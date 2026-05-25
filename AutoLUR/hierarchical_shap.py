import csv
import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import shap.plots.colors as shap_colors
import numpy.random._pickle as np_random_pickle
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from .config import PipelineConfig
from .visualization import format_pollutant_label

logger = logging.getLogger("autolur")

# Avoid loky probing Windows WMIC, which is missing on newer systems.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 1) - 1)))

# Some legacy pickles store the MT19937 class instead of its string name.
_ORIGINAL_BIT_GENERATOR_CTOR = np_random_pickle.__bit_generator_ctor

def _compat_bit_generator_ctor(bit_generator_name: Any = "MT19937") -> Any:
    if isinstance(bit_generator_name, type):
        bit_generator_name = bit_generator_name.__name__
    return _ORIGINAL_BIT_GENERATOR_CTOR(bit_generator_name)

np_random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor

try:
    import shapiq
    SHAPIQ_AVAILABLE = True
except ImportError:
    shapiq = None
    SHAPIQ_AVAILABLE = False


class HierarchicalShapModule:
    """
    层级 SHAP 特征依赖图模块。
    支持 TreeExplainer (Exact) -> LinearExplainer (Exact/Conditional) -> KernelSHAP-IQ (Approximate)
    """

    def __init__(self, output_dir: Path, reports_dir: Path, config: PipelineConfig):
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.panel_dir = self.output_dir / "panels"
        self.single_dir = self.output_dir / "singles"
        self.panel_dir.mkdir(parents=True, exist_ok=True)
        self.single_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        
        # 常量配置
        self.TOP_K = 3
        self.MAX_SAMPLES = config.max_shap_samples if hasattr(config, "max_shap_samples") else 300
        self.RANDOM_SEED = config.random_state if hasattr(config, "random_state") else 42
        self.DPI = config.fig_dpi if hasattr(config, "fig_dpi") else 300
        self.FONT_FAMILY = config.fig_font_family if hasattr(config, "fig_font_family") else "Times New Roman"
        self.SCATTER_CMAP = shap_colors.red_blue
        self.ADDITIVE_AUDIT_TOL = 1e-8
        
        self.LINEAR_EXACT_TYPES = {"LinearRegression", "Ridge", "Lasso", "ElasticNet", "BayesianRidge"}
        self.PLS_TYPES = {"PLSRegression"}
        
        self.SHAPIQ_BUDGET = 512
        self.SHAPIQ_BACKGROUND_SIZE = 50
        self.KERNEL_IQ_INDEX = "k-SII"
        self.KERNEL_IQ_MAX_ORDER = 2

    def sanitize_filename(self, text: str) -> str:
        buf = []
        for ch in text:
            buf.append(ch if ch.isalnum() or ch in "._-" else "_")
        return "".join(buf).strip("_")

    def scalar_predict(self, model: Any, x_frame: pd.DataFrame) -> np.ndarray:
        pred = np.asarray(model.predict(x_frame), dtype=float)
        return pred.reshape(-1)

    def is_tree_explainer_supported(self, model: Any, x_frame: pd.DataFrame) -> tuple[bool, str]:
        try:
            explainer = shap.TreeExplainer(model)
            probe = x_frame.iloc[: min(8, len(x_frame))]
            vals = explainer.shap_interaction_values(probe)
            arr = np.asarray(vals[0] if isinstance(vals, list) else vals, dtype=float)
            if arr.ndim != 3:
                return False, f"unexpected ndim={arr.ndim}"
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def build_tree_bundle(self, model: Any, x_frame: pd.DataFrame) -> dict[str, Any]:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_interaction_values(x_frame)
        arr = np.asarray(vals[0] if isinstance(vals, list) else vals, dtype=float)
        shap_values = arr.sum(axis=2)
        return {
            "supported": True,
            "solver": "tree_exact",
            "exact_status": "Exact",
            "assumptions": "TreeExplainer exact interaction on model-native tree structure",
            "reason": "ok",
            "interaction_values": arr,
            "shap_values": shap_values,
            "base_value": float(np.asarray(explainer.expected_value, dtype=float).reshape(-1)[0]),
            "audit_max_abs_error": 0.0,
            "audit_mean_abs_error": 0.0,
        }

    def build_additive_diagonal_bundle(
        self,
        model: Any,
        x_frame: pd.DataFrame,
        *,
        exact_status: str,
        solver: str,
        assumptions: str,
    ) -> dict[str, Any]:
        x_numeric = x_frame.astype(float).copy()
        n_samples, n_features = x_numeric.shape
        mean_series = x_numeric.mean(axis=0)
        base_frame = pd.DataFrame([mean_series.to_dict()], columns=x_numeric.columns)
        base_value = float(self.scalar_predict(model, base_frame)[0])

        shap_values = np.zeros((n_samples, n_features), dtype=float)
        mean_arr = mean_series.to_numpy(dtype=float)
        for feat_idx, feat_name in enumerate(x_numeric.columns):
            x_one = pd.DataFrame(np.repeat(mean_arr.reshape(1, -1), n_samples, axis=0), columns=x_numeric.columns)
            x_one.loc[:, feat_name] = x_numeric[feat_name].to_numpy(dtype=float)
            pred_one = self.scalar_predict(model, x_one)
            shap_values[:, feat_idx] = pred_one - base_value

        full_pred = self.scalar_predict(model, x_numeric)
        recon = base_value + shap_values.sum(axis=1)
        abs_err = np.abs(full_pred - recon)
        supported = float(abs_err.max()) <= self.ADDITIVE_AUDIT_TOL

        interaction_values = np.zeros((n_samples, n_features, n_features), dtype=float)
        diag_idx = np.arange(n_features)
        interaction_values[:, diag_idx, diag_idx] = shap_values

        return {
            "supported": supported,
            "solver": solver,
            "exact_status": exact_status if supported else "Not exact-tractable",
            "assumptions": assumptions,
            "reason": "ok" if supported else f"additive audit failed: max_err={float(abs_err.max()):.6e}",
            "interaction_values": interaction_values if supported else None,
            "shap_values": shap_values if supported else None,
            "base_value": base_value,
            "audit_max_abs_error": float(abs_err.max()),
            "audit_mean_abs_error": float(abs_err.mean()),
        }

    def build_kernel_iq_bundle(self, model: Any, x_frame: pd.DataFrame) -> dict[str, Any]:
        if not SHAPIQ_AVAILABLE:
            return {
                "supported": False,
                "solver": "kernel_iq_unavailable",
                "exact_status": "Not exact-tractable",
                "assumptions": "shapiq is not installed",
                "reason": "ImportError: shapiq",
                "interaction_values": None,
                "shap_values": None,
                "base_value": math.nan,
                "audit_max_abs_error": math.nan,
                "audit_mean_abs_error": math.nan,
            }

        x_numeric = x_frame.astype(float).copy()
        n_samples, n_features = x_numeric.shape
        feature_names = list(x_numeric.columns)
        x_array = x_numeric.to_numpy(dtype=float)

        bg_size = min(self.SHAPIQ_BACKGROUND_SIZE, n_samples)
        if bg_size < n_samples:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=bg_size, n_init=10, random_state=self.RANDOM_SEED)
            km.fit(x_array)
            background = km.cluster_centers_
        else:
            background = x_array

        def _predict_np(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            df = pd.DataFrame(arr, columns=feature_names)
            return np.asarray(model.predict(df), dtype=float).reshape(-1)

        explainer = shapiq.TabularExplainer(
            model=_predict_np,
            data=background,
            index=self.KERNEL_IQ_INDEX,
            max_order=self.KERNEL_IQ_MAX_ORDER,
            approximator="regression",
            imputer="marginal",
            random_state=self.RANDOM_SEED,
        )

        shap_values = np.zeros((n_samples, n_features), dtype=float)
        interaction_values = np.zeros((n_samples, n_features, n_features), dtype=float)
        base_values = np.zeros(n_samples, dtype=float)

        for i in range(n_samples):
            iv = explainer.explain(x=x_array[i], budget=self.SHAPIQ_BUDGET)
            base_values[i] = float(iv.baseline_value)

            try:
                iterator = iv.dict_values.items()
            except AttributeError:
                iterator = [(k, iv[k]) for k in iv.interaction_lookup]

            for coalition, value in iterator:
                if len(coalition) == 1:
                    shap_values[i, coalition[0]] = float(value)
                elif len(coalition) == 2:
                    a, b = coalition
                    interaction_values[i, a, b] = float(value)
                    interaction_values[i, b, a] = float(value)

            for j in range(n_features):
                interaction_values[i, j, j] = shap_values[i, j]

        base_value = float(np.mean(base_values))
        full_pred = _predict_np(x_array)
        recon = base_value + shap_values.sum(axis=1)
        abs_err = np.abs(full_pred - recon)

        return {
            "supported": True,
            "solver": "kernel_shap_iq",
            "exact_status": "Approximate",
            "assumptions": (
                f"KernelSHAP-IQ with index={self.KERNEL_IQ_INDEX}, max_order={self.KERNEL_IQ_MAX_ORDER}, "
                f"budget={self.SHAPIQ_BUDGET}, background_size={bg_size}"
            ),
            "reason": "shapiq KernelSHAP-IQ fallback",
            "interaction_values": interaction_values,
            "shap_values": shap_values,
            "base_value": base_value,
            "audit_max_abs_error": float(abs_err.max()),
            "audit_mean_abs_error": float(abs_err.mean()),
        }

    def resolve_hierarchical_bundle(self, model: Any, model_type: str, x_frame: pd.DataFrame) -> dict[str, Any]:
        tree_ok, tree_reason = self.is_tree_explainer_supported(model, x_frame)
        if tree_ok:
            return self.build_tree_bundle(model, x_frame)

        if model_type in self.LINEAR_EXACT_TYPES:
            bundle = self.build_additive_diagonal_bundle(
                model, x_frame,
                exact_status="Exact",
                solver="linear_diagonal_exact",
                assumptions="Affine additive model with empirical-mean background",
            )
            if bundle["supported"]:
                return bundle
            additive_reason = str(bundle["reason"])
            bundle = self.build_kernel_iq_bundle(model, x_frame)
            bundle["reason"] = f"{tree_reason}; {additive_reason}; {bundle['reason']}"
            return bundle

        if model_type in self.PLS_TYPES:
            bundle = self.build_additive_diagonal_bundle(
                model, x_frame,
                exact_status="Exact under assumptions",
                solver="pls_linear_reconstruction_exact",
                assumptions="PLS prediction treated as affine on selected transformed feature space",
            )
            if bundle["supported"]:
                return bundle
            additive_reason = str(bundle["reason"])
            bundle = self.build_kernel_iq_bundle(model, x_frame)
            bundle["reason"] = f"{tree_reason}; {additive_reason}; {bundle['reason']}"
            return bundle

        bundle = self.build_kernel_iq_bundle(model, x_frame)
        bundle["reason"] = f"{tree_reason}; {bundle['reason']}"
        return bundle

    def coverage_bucket(self, exact_status: str) -> str:
        if exact_status == "Exact":
            return "strict_exact"
        if exact_status == "Exact under assumptions":
            return "conditional_exact"
        if exact_status == "Approximate":
            return "approximate"
        return "unsupported"

    def compute_and_plot(self, dataset: str, submodels: List[Dict], weights: np.ndarray, x_raw: pd.DataFrame, full_versions: Dict[str, Dict]):
        logger.info("开始计算 %s 的层级 SHAP 依赖图...", dataset)
        
        if len(x_raw) > self.MAX_SAMPLES:
            rng = np.random.RandomState(self.RANDOM_SEED)
            sample_idx = np.sort(rng.choice(x_raw.index.values, size=self.MAX_SAMPLES, replace=False))
        else:
            sample_idx = x_raw.index.values

        feature_names = list(x_raw.columns)
        index_map = {name: i for i, name in enumerate(feature_names)}
        x_display = full_versions["Raw"]["X_full"].loc[sample_idx, feature_names].to_numpy(dtype=float)
        n_samples = len(sample_idx)
        n_features = len(feature_names)

        interaction_sum = np.zeros((n_samples, n_features, n_features), dtype=float)
        shap_sum = np.zeros((n_samples, n_features), dtype=float)

        strict_exact_weight = 0.0
        conditional_exact_weight = 0.0
        approximate_weight = 0.0
        audit_rows: list[dict[str, Any]] = []

        for idx, submodel in enumerate(submodels, start=1):
            weight = float(weights[idx - 1])
            model_type = str(submodel.get("model_name", "Unknown"))
            data_version = str(submodel.get("data_version", "Raw"))
            features = list(submodel.get("features", []))

            row = {
                "dataset": dataset,
                "submodel_index": idx,
                "rank": submodel.get("rank", idx),
                "combo_key": str(submodel.get("combo_key", "")),
                "model_name": model_type,
                "data_version": data_version,
                "weight": f"{weight:.10f}",
                "feature_count": len(features),
                "solver": "",
                "exact_status": "",
                "coverage_bucket": "",
                "supported_for_aggregation": 0,
                "assumptions": "",
                "reason": "",
                "audit_max_abs_error": "",
                "audit_mean_abs_error": "",
            }

            if weight <= 0:
                row["solver"] = "skipped_zero_weight"
                row["exact_status"] = "Not exact-tractable"
                row["coverage_bucket"] = "unsupported"
                row["reason"] = "zero ensemble weight"
                audit_rows.append(row)
                continue

            try:
                model = submodel["estimator"]
                class_name = type(model).__name__
                version_df = full_versions[data_version]["X_full"]
                x_frame = version_df.loc[sample_idx, features]
                bundle = self.resolve_hierarchical_bundle(model, class_name, x_frame)
            except Exception as exc:
                row["solver"] = "submodel_load_or_compute_failed"
                row["exact_status"] = "Not exact-tractable"
                row["coverage_bucket"] = "unsupported"
                row["reason"] = f"{type(exc).__name__}: {exc}"
                audit_rows.append(row)
                continue

            row["solver"] = str(bundle["solver"])
            row["exact_status"] = str(bundle["exact_status"])
            row["coverage_bucket"] = self.coverage_bucket(row["exact_status"])
            row["supported_for_aggregation"] = int(bool(bundle["supported"]))
            row["assumptions"] = str(bundle["assumptions"])
            row["reason"] = str(bundle["reason"])
            if np.isfinite(bundle["audit_max_abs_error"]):
                row["audit_max_abs_error"] = f"{bundle['audit_max_abs_error']:.10e}"
            if np.isfinite(bundle["audit_mean_abs_error"]):
                row["audit_mean_abs_error"] = f"{bundle['audit_mean_abs_error']:.10e}"

            if not bundle["supported"]:
                audit_rows.append(row)
                continue

            local_interaction = np.asarray(bundle["interaction_values"], dtype=float)
            local_shap = np.asarray(bundle["shap_values"], dtype=float)
            mapped_idx = np.asarray([index_map[f] for f in features], dtype=int)
            interaction_sum[:, mapped_idx[:, None], mapped_idx] += weight * local_interaction
            shap_sum[:, mapped_idx] += weight * local_shap

            if row["exact_status"] == "Exact":
                strict_exact_weight += weight
            elif row["exact_status"] == "Exact under assumptions":
                conditional_exact_weight += weight
            elif row["exact_status"] == "Approximate":
                approximate_weight += weight

            audit_rows.append(row)

        supported_weight = strict_exact_weight + conditional_exact_weight + approximate_weight
        if supported_weight <= 0:
            logger.warning(f"{dataset}: 没有可支持层级 SHAP 的子模型 (或权重均为 0)")
            return

        total_weight = float(sum(weights)) if len(weights) > 0 else 0.0
        norm_weight = total_weight if total_weight > 0 else supported_weight
        interaction_final = interaction_sum / norm_weight
        shap_final = shap_sum / norm_weight

        result = {
            "dataset": dataset,
            "feature_names": feature_names,
            "x_display": x_display,
            "interaction_values": interaction_final,
            "shap_values": shap_final,
            "n_samples": n_samples,
            "n_features": n_features,
            "submodel_count": len(submodels),
            "supported_submodel_count": sum(int(r["supported_for_aggregation"]) for r in audit_rows),
            "strict_exact_submodel_count": sum(r["exact_status"] == "Exact" for r in audit_rows),
            "conditional_exact_submodel_count": sum(r["exact_status"] == "Exact under assumptions" for r in audit_rows),
            "approximate_submodel_count": sum(r["exact_status"] == "Approximate" for r in audit_rows),
            "total_weight": total_weight,
            "supported_weight": supported_weight,
            "strict_exact_weight": strict_exact_weight,
            "conditional_exact_weight": conditional_exact_weight,
            "approximate_weight": approximate_weight,
            "coverage_supported": supported_weight / total_weight if total_weight > 0 else 0.0,
            "coverage_strict": strict_exact_weight / total_weight if total_weight > 0 else 0.0,
            "coverage_conditional": conditional_exact_weight / total_weight if total_weight > 0 else 0.0,
            "coverage_approximate": approximate_weight / total_weight if total_weight > 0 else 0.0,
        }
        
        self.render_one_dataset(result)
        
        # Save audit and summary CSVs
        self._write_csv(self.reports_dir / f"{dataset}_hierarchical_submodel_audit.csv", audit_rows, list(audit_rows[0].keys()))

        model_summary = [{
            "dataset": dataset,
            "coverage_supported": f"{result['coverage_supported']:.10f}",
            "coverage_strict": f"{result['coverage_strict']:.10f}",
            "coverage_conditional": f"{result['coverage_conditional']:.10f}",
            "coverage_approximate": f"{result['coverage_approximate']:.10f}",
            "supported_weight": f"{result['supported_weight']:.10f}",
            "strict_exact_weight": f"{result['strict_exact_weight']:.10f}",
            "conditional_exact_weight": f"{result['conditional_exact_weight']:.10f}",
            "approximate_weight": f"{result['approximate_weight']:.10f}",
            "total_weight": f"{result['total_weight']:.10f}",
            "supported_submodel_count": result["supported_submodel_count"],
            "strict_exact_submodel_count": result["strict_exact_submodel_count"],
            "conditional_exact_submodel_count": result["conditional_exact_submodel_count"],
            "approximate_submodel_count": result["approximate_submodel_count"],
            "submodel_count": result["submodel_count"],
            "n_samples": result["n_samples"],
            "n_features": result["n_features"],
            "status": "ok",
        }]
        self._write_csv(self.reports_dir / f"{dataset}_hierarchical_model_summary.csv", model_summary, list(model_summary[0].keys()))
        logger.info("%s 的层级 SHAP 分析完成。", dataset)


    def _choose_partner_index(self, feature_idx: int, top_indices: np.ndarray, interaction_strength: np.ndarray, x_display: np.ndarray) -> int:
        row = interaction_strength[feature_idx].copy()
        row[feature_idx] = -np.inf
        if np.isfinite(np.nanmax(row)) and float(np.nanmax(row)) > 0:
            return int(np.nanargmax(row))

        candidates = [int(i) for i in top_indices if int(i) != int(feature_idx)]
        if not candidates:
            return int(feature_idx)
        x0 = x_display[:, feature_idx]
        best = candidates[0]
        best_score = -1.0
        for candidate in candidates:
            xc = x_display[:, candidate]
            mask = np.isfinite(x0) & np.isfinite(xc)
            if mask.sum() < 3:
                continue
            score = abs(float(np.corrcoef(x0[mask], xc[mask])[0, 1]))
            if np.isfinite(score) and score > best_score:
                best_score = score
                best = candidate
        return int(best)

    def build_axis(self, ax: plt.Axes, feature_name: str, interaction_name: str, x_feature: np.ndarray, y_shap: np.ndarray, color_feature: np.ndarray, importance_value: float, interaction_strength: float, coverage_supported: float, coverage_strict: float, coverage_conditional: float, coverage_approximate: float, rank: int) -> None:
        mask = np.isfinite(x_feature) & np.isfinite(y_shap) & np.isfinite(color_feature)
        x = x_feature[mask]
        y = y_shap[mask]
        c = color_feature[mask]

        if x.size == 0:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
            return

        c_min = np.nanpercentile(c, 5)
        c_max = np.nanpercentile(c, 95)
        if not np.isfinite(c_min) or not np.isfinite(c_max) or math.isclose(float(c_min), float(c_max)):
            c_min = np.nanmin(c)
            c_max = np.nanmax(c)
        if math.isclose(float(c_min), float(c_max)):
            c_min = float(c_min) - 1.0
            c_max = float(c_max) + 1.0

        norm = Normalize(vmin=float(c_min), vmax=float(c_max))
        scat = ax.scatter(
            x, y, c=c, cmap=self.SCATTER_CMAP, norm=norm, s=35, alpha=0.82, edgecolors="white", linewidths=0.45,
        )
        ax.axhline(0.0, color="#3b3b3b", linestyle="--", linewidth=1.0, alpha=0.75)

        ax.set_xlabel(feature_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("SHAP value", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        cbar = plt.colorbar(scat, ax=ax, fraction=0.05, pad=0.03)
        cbar.set_label(interaction_name, fontsize=10, fontweight="bold")
        cbar.ax.tick_params(labelsize=9)

    def render_one_dataset(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        dataset = result["dataset"]
        dataset_label = format_pollutant_label(dataset)
        feature_names = result["feature_names"]
        x_display = np.asarray(result["x_display"], dtype=float)
        shap_values = np.asarray(result["shap_values"], dtype=float)
        interaction_values = np.asarray(result["interaction_values"], dtype=float)
        coverage_supported = float(result["coverage_supported"])
        coverage_strict = float(result["coverage_strict"])
        coverage_conditional = float(result["coverage_conditional"])
        coverage_approximate = float(result["coverage_approximate"])

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:self.TOP_K]
        interaction_strength_matrix = np.mean(np.abs(interaction_values), axis=0)
        np.fill_diagonal(interaction_strength_matrix, -np.inf)
        records: list[dict[str, Any]] = []

        with plt.rc_context(
            {
                "font.family": self.FONT_FAMILY,
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        ):
            fig, axes = plt.subplots(1, self.TOP_K, figsize=(16.8, 4.9), dpi=self.DPI, constrained_layout=False)
            if self.TOP_K == 1:
                axes = [axes]

            for rank, (ax, feature_idx) in enumerate(zip(axes, top_indices), start=1):
                partner_idx = self._choose_partner_index(
                    int(feature_idx),
                    top_indices=top_indices,
                    interaction_strength=interaction_strength_matrix,
                    x_display=x_display,
                )
                feature_name = feature_names[int(feature_idx)]
                partner_name = feature_names[int(partner_idx)]
                importance_value = float(mean_abs_shap[int(feature_idx)])
                interaction_strength = float(
                    np.mean(np.abs(interaction_values[:, int(feature_idx), int(partner_idx)]))
                )

                self.build_axis(
                    ax=ax, feature_name=feature_name, interaction_name=partner_name,
                    x_feature=x_display[:, int(feature_idx)], y_shap=shap_values[:, int(feature_idx)], color_feature=x_display[:, int(partner_idx)],
                    importance_value=importance_value, interaction_strength=interaction_strength,
                    coverage_supported=coverage_supported, coverage_strict=coverage_strict, coverage_conditional=coverage_conditional, coverage_approximate=coverage_approximate, rank=rank,
                )
                ax.set_title("")

                single_fig, single_ax = plt.subplots(1, 1, figsize=(5.7, 4.9), dpi=self.DPI, constrained_layout=True)
                self.build_axis(
                    ax=single_ax, feature_name=feature_name, interaction_name=partner_name,
                    x_feature=x_display[:, int(feature_idx)], y_shap=shap_values[:, int(feature_idx)], color_feature=x_display[:, int(partner_idx)],
                    importance_value=importance_value, interaction_strength=interaction_strength,
                    coverage_supported=coverage_supported, coverage_strict=coverage_strict, coverage_conditional=coverage_conditional, coverage_approximate=coverage_approximate, rank=rank,
                )
                single_ax.set_title(dataset_label, fontsize=14, fontweight="bold")
                single_path = self.single_dir / f"{self.sanitize_filename(dataset)}__TOP{rank}__{self.sanitize_filename(feature_name)}__hierarchical.png"
                single_fig.savefig(single_path, dpi=self.DPI, bbox_inches="tight", facecolor="white")
                plt.close(single_fig)

                records.append(
                    {
                        "dataset": dataset,
                        "rank": rank,
                        "feature_name": feature_name,
                        "interaction_partner": partner_name,
                        "mean_abs_shap": f"{importance_value:.10f}",
                        "mean_abs_interaction": f"{interaction_strength:.10f}",
                        "coverage_supported": f"{coverage_supported:.10f}",
                        "coverage_strict": f"{coverage_strict:.10f}",
                        "coverage_conditional": f"{coverage_conditional:.10f}",
                        "coverage_approximate": f"{coverage_approximate:.10f}",
                        "supported_submodel_count": result["supported_submodel_count"],
                        "strict_exact_submodel_count": result["strict_exact_submodel_count"],
                        "conditional_exact_submodel_count": result["conditional_exact_submodel_count"],
                        "approximate_submodel_count": result["approximate_submodel_count"],
                        "submodel_count": result["submodel_count"],
                        "single_plot": str(single_path),
                    }
                )

            panel_path = self.panel_dir / f"{self.sanitize_filename(dataset)}_hierarchical_top{self.TOP_K}_dependence.png"
            fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.84, wspace=0.28)
            fig.suptitle(dataset_label, fontsize=15, fontweight="bold", y=0.965)
            fig.savefig(panel_path, dpi=self.DPI, facecolor="white")
            plt.close(fig)

        for row in records:
            row["panel_plot"] = str(panel_path)
            
        self._write_csv(self.reports_dir / f"{dataset}_hierarchical_top3_features.csv", records, list(records[0].keys()))
        return records

    def _write_csv(self, path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
