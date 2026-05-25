"""
可视化模块（增强版）
融合 LURV1 出版级图表质量 + LURV2 集成模型架构

包含:
  - 散点图: 主散点 + 边际 KDE + 残差面板 + 回归线 + 统计注解框
  - SHAP图: 蜂巢图 + 条形图叠加 + 百分比标注 + 对称轴 + 自适应布局
  - 污染物名称 LaTeX 格式化
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FuncFormatter, MaxNLocator, FormatStrFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import PipelineConfig

logger = logging.getLogger("autolur")

# ── 污染物名称格式化 ──────────────────────────────────────

_TIME_MAP = {
    "ALL": "-All Day",
    "DAY": "-Day",
    "NIGHT": "-Night",
}


def format_pollutant_label(raw_name: str) -> str:
    """将 O3ALL → $O_3$-All Day 等 LaTeX 格式标签"""
    clean = raw_name.replace("PM2_5", "PM2.5")
    if clean.endswith("_Ensemble"):
        clean = clean[:-9]
    base, suffix = clean, ""
    for key, text in _TIME_MAP.items():
        if clean.endswith(key):
            base = clean[: -len(key)]
            suffix = text
            break
    m = re.match(r"([A-Za-z]+)([\d.]+)", base)
    if m:
        latex = f"{m.group(1)}$_{{{m.group(2)}}}$"
    else:
        latex = base
    return f"{latex}{suffix}"


# ── 辅助 ─────────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


# ── 散点图配色方案 ──

C_FIT = "#3C5488"     # NPG 蓝 - Model Fitting
C_CV = "#E64B35"      # NPG 橙红 - Cross-Validation
C_GRID = "#CCCCCC"
C_OBS = "#00A087"     # NPG 青绿 - Observed


# ═══════════════════════════════════════════════════════════
# 可视化模块
# ═══════════════════════════════════════════════════════════

class VisualizationModule:
    """出版级可视化：散点残差图 + SHAP 特征重要性图"""

    def __init__(self, output_dir: Path, config: Optional[PipelineConfig] = None):
        self.output_dir = Path(output_dir)
        _ensure_dir(self.output_dir)
        self.config = config or PipelineConfig()

    # ══════════════════════════════════════════════════════
    # 散点图 + 残差图 + 边际 KDE（出版级）
    # ══════════════════════════════════════════════════════

    def plot_scatter_residual(
        self,
        y_true: np.ndarray,
        y_fit: np.ndarray,
        y_cv: np.ndarray,
        file_name: str,
        dataset_name: str = "",
        fit_metrics: Optional[Dict] = None,
        cv_metrics: Optional[Dict] = None,
    ) -> Path:
        """
        出版级散点残差图。

        布局 (3×2 GridSpec):
          [0,0] 顶部边际 KDE（Observed 分布）
          [0,1] 空白
          [1,0] 主散点图 (Fit + CV + 1:1线 + 回归线)
          [1,1] 右侧边际 KDE（Predicted 分布）
          [2,0] 残差图 + 零线
          [2,1] 残差边际 KDE
        """
        y_true = np.asarray(y_true, dtype=float)
        y_fit = np.asarray(y_fit, dtype=float)
        y_cv = np.asarray(y_cv, dtype=float)

        # 计算指标
        if fit_metrics is None:
            fit_metrics = _metric_dict(y_true, y_fit)
        if cv_metrics is None:
            cv_metrics = _metric_dict(y_true, y_cv)

        # 坐标范围
        all_vals = np.concatenate([y_true, y_fit, y_cv])
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        pad = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
        limits = [vmin - pad, vmax + pad]

        # 污染物标签
        pollutant = format_pollutant_label(dataset_name) if dataset_name else ""
        xlabel = f"Observed {pollutant} (μg/m³)" if pollutant else "Observed"
        ylabel = f"Predicted {pollutant} (μg/m³)" if pollutant else "Predicted"

        # 样式
        style = self._pub_style()

        with plt.rc_context(style):
            fig = plt.figure(
                figsize=self.config.scatter_figsize,
                dpi=self.config.fig_dpi,
            )
            gs = gridspec.GridSpec(
                3, 2,
                width_ratios=[4, 0.32],
                height_ratios=[0.32, 4.0, 0.85],
                wspace=0.05,
                hspace=0.05,
            )
            fig.subplots_adjust(left=0.15, right=0.95, bottom=0.08, top=0.97)

            # ── 顶部 KDE ──
            ax_top = fig.add_subplot(gs[0, 0])
            self._kde_strip(ax_top, y_true, C_OBS, axis="x", label="Observed")
            self._kde_strip(ax_top, y_fit, C_FIT, axis="x", label="Model Fitting")
            ax_top.set_xlim(limits)
            self._hide_all_spines(ax_top)

            # ── 右侧 KDE ──
            ax_right = fig.add_subplot(gs[1, 1])
            if np.nanstd(y_true) > 0:
                sns.kdeplot(y=y_true, color=C_OBS, fill=True, alpha=0.18, ax=ax_right, lw=1.4)
            if np.nanstd(y_cv) > 0:
                sns.kdeplot(y=y_cv, color=C_CV, fill=True, alpha=0.25, ax=ax_right, lw=1.5)
            ax_right.set_ylim(limits)
            self._hide_all_spines(ax_right)

            # ── 主散点图 ──
            ax_main = fig.add_subplot(gs[1, 0])
            ax_main.plot(limits, limits, ls="--", c="black", alpha=0.4, lw=1.5, zorder=1)

            ax_main.scatter(
                y_true, y_fit, c=C_FIT, alpha=0.45, s=40,
                edgecolor="white", lw=0.5, label="Model Fitting", zorder=2,
            )
            ax_main.scatter(
                y_true, y_cv, c=C_CV, alpha=0.75, s=50,
                edgecolor="white", lw=0.5, label="Cross-Validation", zorder=3,
            )
            # 回归线 + 95% CI
            if len(y_true) >= 3:
                sns.regplot(
                    x=y_true, y=y_fit, ax=ax_main, scatter=False,
                    color=C_FIT, truncate=False, ci=95,
                    line_kws={"lw": 2.2, "alpha": 0.75},
                )
                sns.regplot(
                    x=y_true, y=y_cv, ax=ax_main, scatter=False,
                    color=C_CV, truncate=False, ci=95,
                    line_kws={"lw": 2.5, "alpha": 0.8},
                )

            ax_main.set_xlim(limits)
            ax_main.set_ylim(limits)
            ax_main.tick_params(labelbottom=False)
            ax_main.set_xlabel("")
            ax_main.set_ylabel(ylabel, fontsize=self.config.fig_font_size, fontweight="bold")
            ax_main.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_main.grid(True, linestyle=":", alpha=0.6)

            # 图例
            legend = ax_main.legend(
                loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2,
                frameon=True, fontsize=self.config.fig_font_size,
                markerscale=1.6, handletextpad=0.08,
                columnspacing=0.5, borderaxespad=0.35,
            )
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_edgecolor("#CCCCCC")

            # 统计注解框（同时显示 Fit 和 CV 指标）
            stats_text = (
                f"Model Fitting:\n"
                f"  R² = {fit_metrics['R2']:.3f}\n"
                f"  RMSE = {fit_metrics['RMSE']:.2f}\n"
                f"  MAE = {fit_metrics['MAE']:.2f}\n"
                f"Cross-Validation:\n"
                f"  R² = {cv_metrics['R2']:.3f}\n"
                f"  RMSE = {cv_metrics['RMSE']:.2f}\n"
                f"  MAE = {cv_metrics['MAE']:.2f}"
            )
            stats_box = AnchoredText(
                stats_text, loc="lower right",
                prop=dict(size=self.config.fig_font_size - 4, family=self.config.fig_font_family),
                frameon=True, borderpad=0.65, pad=0.45,
            )
            stats_box.patch.set_facecolor("white")
            stats_box.patch.set_alpha(0.92)
            stats_box.patch.set_edgecolor("#CCCCCC")
            ax_main.add_artist(stats_box)

            # ── 残差图 ──
            ax_resid = fig.add_subplot(gs[2, 0])
            resid_fit = y_fit - y_true
            resid_cv = y_cv - y_true
            ax_resid.axhline(0, color="black", lw=1.2, alpha=0.8)
            ax_resid.scatter(
                y_true, resid_fit, c=C_FIT, alpha=0.45, s=30,
                edgecolor="white", lw=0.3, label="Fitting",
            )
            ax_resid.scatter(
                y_true, resid_cv, c=C_CV, alpha=0.75, s=40,
                edgecolor="white", lw=0.3, label="CV",
            )
            ax_resid.set_xlim(limits)
            ax_resid.set_xlabel(xlabel, fontsize=self.config.fig_font_size, fontweight="bold")
            ax_resid.set_ylabel("Residuals", fontsize=self.config.fig_font_size, fontweight="bold")
            ax_resid.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax_resid.grid(True, linestyle=":", alpha=0.6)

            # 残差 KDE
            ax_resid_kde = fig.add_subplot(gs[2, 1])
            if np.nanstd(resid_fit) > 0:
                sns.kdeplot(y=resid_fit, color=C_FIT, fill=True, alpha=0.25, ax=ax_resid_kde, lw=1.5)
            if np.nanstd(resid_cv) > 0:
                sns.kdeplot(y=resid_cv, color=C_CV, fill=True, alpha=0.25, ax=ax_resid_kde, lw=1.5)
            ax_resid_kde.set_ylim(ax_resid.get_ylim())
            ax_resid_kde.set_xlabel("")
            ax_resid_kde.set_ylabel("")
            self._hide_all_spines(ax_resid_kde)

        out = self.output_dir / file_name
        fig.savefig(out, dpi=self.config.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("散点残差图已保存: %s", out)
        return out

    # ══════════════════════════════════════════════════════
    # SHAP 特征重要性图（蜂巢 + 条形 + 百分比标注）
    # ══════════════════════════════════════════════════════

    def plot_shap_for_weighted_models(
        self,
        submodels: List[Dict[str, Any]],
        weights: np.ndarray,
        full_versions: Dict[str, Dict[str, pd.DataFrame]],
        file_name: str,
        max_samples: int = 300,
        dataset_name: str = "",
    ) -> Path:
        """
        计算加权 SHAP 值并绘制出版级图表。

        使用 TreeExplainer / LinearExplainer / KernelExplainer
        自动适配不同模型类型。
        """
        raw_df = full_versions["Raw"]["X_full"]

        # 采样
        if len(raw_df) > max_samples:
            rng = np.random.RandomState(42)
            sample_idx = np.sort(rng.choice(raw_df.index.values, size=max_samples, replace=False))
        else:
            sample_idx = raw_df.index.values
        sample_df = raw_df.loc[sample_idx]

        feature_list = list(sample_df.columns)
        feature_to_idx = {f: i for i, f in enumerate(feature_list)}
        agg = np.zeros((len(sample_df), len(feature_list)), dtype=float)

        # 模型类型集合
        tree_types = {
            "XGBRegressor", "LGBMRegressor", "CatBoostRegressor",
            "RandomForestRegressor", "ExtraTreesRegressor",
            "GradientBoostingRegressor", "DecisionTreeRegressor",
        }
        linear_types = {
            "Ridge", "Lasso", "ElasticNet", "LinearRegression",
            "BayesianRidge", "PLSRegression", "QuantileRegressor",
        }

        for i, item in enumerate(submodels):
            model = item["estimator"]
            version = item["data_version"]
            feats = item["features"]
            model_type = type(model).__name__

            X_version = full_versions[version]["X_full"].loc[sample_idx, feats]
            try:
                if model_type in tree_types:
                    explainer = shap.TreeExplainer(model)
                    vals = explainer.shap_values(X_version)
                elif model_type in linear_types:
                    explainer = shap.LinearExplainer(model, X_version)
                    vals = explainer.shap_values(X_version)
                else:
                    bg = shap.sample(X_version, min(80, len(X_version)))
                    explainer = shap.KernelExplainer(model.predict, bg)
                    vals = explainer.shap_values(X_version)

                if isinstance(vals, list):
                    vals = vals[0]
                for j, f in enumerate(feats):
                    agg[:, feature_to_idx[f]] += float(weights[i]) * vals[:, j]
            except Exception:
                logger.exception("SHAP 计算失败，跳过子模型 model=%s version=%s", model_type, version)

        max_display = min(self.config.shap_max_display, len(feature_list))
        out = self._draw_shap_dual_axis(
            agg, sample_df, feature_list, max_display,
            dataset_name=dataset_name, file_name=file_name,
        )
        return out

    # ──────────────────────────────────────────────────────
    # SHAP 双轴绘图核心（蜂巢 + 半透明条形 + 百分比标注）
    # ──────────────────────────────────────────────────────

    def _draw_shap_dual_axis(
        self,
        shap_values: np.ndarray,
        X_display: pd.DataFrame,
        feature_names: List[str],
        max_display: int,
        dataset_name: str = "",
        file_name: str = "shap.png",
    ) -> Path:
        style = self._pub_style()
        style.update({
            "font.size": 19,
            "axes.labelsize": 19,
            "axes.titlesize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        })

        with plt.rc_context(style):
            fig, ax1 = plt.subplots(
                figsize=self.config.shap_figsize,
                dpi=self.config.fig_dpi,
            )
            max_display = min(max_display, len(feature_names))

            # ── 蜂巢图 ──
            plt.sca(ax1)
            shap.summary_plot(
                shap_values, X_display,
                feature_names=feature_names,
                plot_type="dot",
                max_display=max_display,
                show=False,
                color_bar=True,
            )
            ax1 = plt.gca()
            ax1.set_position([0.24, 0.18, 0.60, 0.72])

            # 对称 x 轴
            xmin, xmax = ax1.get_xlim()
            m_data = max(abs(xmin), abs(xmax))
            if np.isfinite(m_data) and m_data > 0:
                m_tick = m_data
                m_lim = m_tick * 1.12
                ax1.set_xlim(-m_lim, m_lim)
                ax1.set_xticks([-m_tick, -m_tick / 2, 0, m_tick / 2, m_tick])
                ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            # 保存蜂巢 y 轴信息
            bee_yticks = ax1.get_yticks()
            bee_ylabels = [t.get_text() for t in ax1.get_yticklabels()]

            # ── 条形图叠加 ──
            ax2 = ax1.twiny()
            ax2.set_position(ax1.get_position())
            plt.sca(ax2)
            shap.summary_plot(
                shap_values, X_display,
                feature_names=feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False,
                color_bar=False,
            )

            ax2.set_ylim(ax1.get_ylim())
            ax2.set_yticks([])
            ax2.set_ylabel("")
            for bar in ax2.patches:
                bar.set_facecolor("#f2a7b5")
                bar.set_alpha(0.35)
                bar.set_edgecolor("none")
            ax2.set_zorder(0)
            ax1.set_zorder(1)
            ax1.patch.set_visible(False)

            # ── 百分比标注 ──
            sv = np.asarray(shap_values)
            mean_abs_all = np.mean(np.abs(sv), axis=0)
            order = np.argsort(mean_abs_all)[::-1][:max_display]
            mean_abs_top = mean_abs_all[order]
            total_top = float(np.sum(mean_abs_top)) if np.isfinite(np.sum(mean_abs_top)) else 1.0

            # 恢复 y 轴标签
            if bee_ylabels:
                ax1.set_yticks(bee_yticks)
                ax1.set_yticklabels(bee_ylabels)
                for label in ax1.get_yticklabels():
                    label.set_fontsize(20)
                    label.set_fontweight("bold")

            # 在条形上标注数值和百分比
            bx_min, bx_max = ax2.get_xlim()
            bx0 = 0.0 if bx_min <= 0 <= bx_max else bx_min
            bx_text = bx0 + 0.01 * (bx_max - bx_min)

            bars_sorted = sorted(
                ax2.patches,
                key=lambda b: b.get_y() + b.get_height() / 2,
                reverse=True,
            )
            n = min(len(bars_sorted), len(mean_abs_top))
            for i in range(n):
                bar = bars_sorted[i]
                by = bar.get_y() + bar.get_height() / 2
                v = float(mean_abs_top[i])
                pct = 100.0 * v / total_top if total_top > 0 else 0
                ax2.text(
                    bx_text, by, f"{v:.3f}({pct:.1f}%)",
                    va="center", ha="left", fontsize=9, color="black", zorder=3,
                )

            # ── 美化边框 ──
            for spine in ("top", "right", "bottom", "left"):
                ax1.spines[spine].set_visible(True)
                ax1.spines[spine].set_linewidth(1.0)
            ax1.tick_params(axis="y", which="major", length=6, direction="out")
            ax1.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
            ax1.set_axisbelow(True)
            ax1.set_xlabel("SHAP Value (impact on model output)", fontsize=11, fontweight="bold")
            ax1.set_ylabel("")
            ax2.set_xlabel("Mean (|SHAP|)", fontsize=11, fontweight="bold")
            ax2.xaxis.set_label_position("top")
            ax2.xaxis.tick_top()

            # 标题
            pollutant = format_pollutant_label(dataset_name) if dataset_name else "Ensemble"
            plt.title(
                f"{pollutant} SHAP Feature Importance",
                fontsize=12, fontweight="bold", pad=10,
            )
            plt.tight_layout()

        out = self.output_dir / file_name
        fig.savefig(out, dpi=self.config.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP 图已保存: %s", out)
        return out

    # ══════════════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════════════

    def _pub_style(self) -> dict:
        """返回出版级 matplotlib rcParams"""
        fs = self.config.fig_font_size
        return {
            "font.family": self.config.fig_font_family,
            "font.size": fs,
            "axes.labelsize": fs,
            "axes.titlesize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
            "legend.fontsize": fs,
            "mathtext.default": "regular",
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }

    @staticmethod
    def _kde_strip(ax, data, color, axis="x", label=None):
        """绘制单侧 KDE 密度条"""
        data = np.asarray(data, dtype=float)
        if data.size < 2 or np.nanstd(data) == 0:
            return
        if axis == "x":
            sns.kdeplot(data, color=color, fill=True, alpha=0.3, ax=ax, lw=1.5, label=label)
        else:
            sns.kdeplot(y=data, color=color, fill=True, alpha=0.3, ax=ax, lw=1.5, label=label)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.set_xlabel("")

    @staticmethod
    def _hide_all_spines(ax):
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
