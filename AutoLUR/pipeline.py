"""
LUR Pipeline 主编排模块

完整流程:
  1. 数据加载 + 空间聚类划分
  2. 外层 K 折 CV，每折内独立特征工程 + 全组合训练
  3. OOF 矩阵汇总 + Top N 筛选 + Greedy 权重优化
  4. 全量数据重训最终子模型
  5. 外部数据预测 + 出版级可视化 + 层级 SHAP 分析
"""
import ctypes
import gc
import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import PipelineConfig
from .data_module import DataModule
from .ensemble_module import EnsembleModule
from .feature_module import FeatureModule
from .model_registry import ModelRegistry
from .modeling_module import ModelingModule
from .visualization import VisualizationModule
from .hierarchical_shap import HierarchicalShapModule

logger = logging.getLogger("autolur")


# ── 工具函数 ─────────────────────────────────────────────

def _safe_name(name: str) -> str:
    return (
        str(name)
        .replace("\u200b", "").replace("\ufeff", "").strip()
        .replace("/", "_").replace("\\", "_")
        .replace(" ", "_").replace(":", "_")
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


# ── Fold Worker（用于多进程并行） ────────────────────────

@dataclass
class FoldWorkerArgs:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    X: pd.DataFrame
    y: pd.Series
    cache_dir: Path
    config: PipelineConfig


def _fold_worker(args: FoldWorkerArgs) -> Dict[str, Any]:
    """子进程入口：独立创建 Pipeline 实例避免共享状态"""
    pipeline = LURPipeline(config=args.config)
    subsets = pipeline._run_feature_engineering_for_fold(
        cache_dir=args.cache_dir,
        fold_id=args.fold_id,
        train_idx=args.train_idx,
        test_idx=args.test_idx,
        X=args.X,
        y=args.y,
    )
    return pipeline._train_one_fold(
        cache_dir=args.cache_dir,
        fold_id=args.fold_id,
        train_idx=args.train_idx,
        test_idx=args.test_idx,
        X=args.X,
        y=args.y,
        feature_subsets=subsets,
        n_jobs_inner=max(1, int(args.config.n_jobs_inner)),
    )


# ═══════════════════════════════════════════════════════════
# LUR Pipeline
# ═══════════════════════════════════════════════════════════

class LURPipeline:
    """
    AutoLUR 核心编排类。

    公开方法:
      - run_single_file(train_file, predict_file, output_root) → dict
      - run_batch(input_dir, predict_file, output_root) → dict
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        log_level: int = logging.INFO,
    ):
        self.config = config or PipelineConfig()
        logger.setLevel(log_level)
        self.data_module = DataModule(self.config)
        self.feature_module = FeatureModule(self.config)
        self.registry = ModelRegistry()
        self.modeling_module = ModelingModule(self.config, self.registry)
        self.ensemble_module = EnsembleModule(self.config)

    # ══════════════════════════════════════════════════════
    # 缓存路径
    # ══════════════════════════════════════════════════════

    def _fold_feature_cache(self, cache_dir: Path, fold_id: int) -> Path:
        return cache_dir / "feature_engineering" / f"fold_{fold_id:02d}_subsets.json"

    def _fold_model_cache(self, cache_dir: Path, fold_id: int) -> Path:
        return cache_dir / "modeling" / f"fold_{fold_id:02d}_records.pkl"

    def _oof_chunks_dir(self, cache_dir: Path) -> Path:
        return cache_dir / "modeling" / "oof_chunks"

    def _oof_chunk_path(self, cache_dir: Path, fold_id: int) -> Path:
        return self._oof_chunks_dir(cache_dir) / f"fold_{fold_id:02d}.npz"

    def _oof_meta_path(self, cache_dir: Path) -> Path:
        return cache_dir / "modeling" / "oof_meta.json"

    def _cache_exists(self, path: Path) -> bool:
        return self.config.use_cache and path.exists()

    # ══════════════════════════════════════════════════════
    # 并行策略
    # ══════════════════════════════════════════════════════

    def _resolve_parallel_mode(self) -> str:
        mode = str(self.config.parallel_mode).strip().lower()
        allowed = {"off", "conservative", "balanced", "aggressive", "auto"}
        if mode not in allowed:
            logger.warning("未知 parallel_mode=%s，回退 auto", mode)
            mode = "auto"
        return mode

    def _resolve_oof_storage(self) -> str:
        s = str(self.config.oof_storage).strip().lower()
        return s if s in ("memory", "disk") else "memory"

    def _get_total_memory_bytes(self) -> int:
        """跨平台获取物理内存"""
        # Linux / macOS
        if hasattr(os, "sysconf"):
            try:
                total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
                if total > 0:
                    return total
            except Exception:
                pass
        # Windows
        try:
            class _MS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullTotalPhys)
        except Exception:
            pass
        return 0

    def _get_memory_load_ratio(self) -> float:
        try:
            class _MS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return float(ms.dwMemoryLoad) / 100.0
        except Exception:
            pass
        return -1.0

    def _estimate_fold_mem(self, X: pd.DataFrame, y: pd.Series) -> int:
        return int(X.memory_usage(index=True, deep=True).sum() + y.memory_usage(index=True, deep=True))

    def _resolve_fold_workers(self, outer_k: int, X: pd.DataFrame, y: pd.Series) -> int:
        mode = self._resolve_parallel_mode()
        req = int(self.config.n_jobs_fold)
        if req == 1 or mode == "off":
            return 1
        cpus = os.cpu_count() or 1
        usable = max(1, cpus - max(0, self.config.cpu_reserve_cores))
        budget = min(self.config.max_workers_cap, usable)
        if req > 1:
            return min(outer_k, req, budget)
        divisor = {"conservative": 4, "balanced": 3, "aggressive": 2, "auto": 3}.get(mode, 3)
        workers = min(outer_k, max(1, budget // divisor))
        total_mem = self._get_total_memory_bytes()
        if total_mem > 0 and 0.1 <= self.config.mem_util_target <= 0.95:
            per_fold = max(1, self._estimate_fold_mem(X, y))
            workers = min(workers, max(1, int(total_mem * self.config.mem_util_target // per_fold)))
        return max(1, workers)

    def _resolve_inner_workers(self, outer_workers: int) -> int:
        mode = self._resolve_parallel_mode()
        if mode == "off":
            return 1
        cpus = os.cpu_count() or 1
        usable = max(1, cpus - max(0, self.config.cpu_reserve_cores))
        budget = min(self.config.max_workers_cap, usable)
        per_outer = max(1, budget // max(1, outer_workers))
        req = int(self.config.n_jobs_inner)
        if req > 0:
            inner = min(req, per_outer, self.config.max_workers_cap)
        elif mode == "aggressive":
            inner = min(3, per_outer)
        elif mode == "balanced":
            inner = min(2, per_outer)
        else:
            inner = 1
        if self._resolve_oof_storage() != "disk":
            inner = min(inner, 2)
        return max(1, inner)

    # ══════════════════════════════════════════════════════
    # 单 fold 训练
    # ══════════════════════════════════════════════════════

    def _run_feature_engineering_for_fold(
        self, cache_dir: Path, fold_id: int,
        train_idx: np.ndarray, test_idx: np.ndarray,
        X: pd.DataFrame, y: pd.Series,
    ) -> Dict[str, List[str]]:
        path = self._fold_feature_cache(cache_dir, fold_id)
        if self._cache_exists(path):
            logger.info("Fold %s 命中特征缓存", fold_id)
            return _load_json(path)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        versions = self.data_module.fit_transform_versions(X_train, X_test)
        subsets = self.feature_module.run_fold_feature_engineering(versions, y_train)
        _save_json(path, subsets)
        return subsets

    def _train_one_fold(
        self,
        cache_dir: Path, fold_id: int,
        train_idx: np.ndarray, test_idx: np.ndarray,
        X: pd.DataFrame, y: pd.Series,
        feature_subsets: Dict[str, List[str]],
        n_jobs_inner: int = 1,
    ) -> Dict[str, Any]:
        out_path = self._fold_model_cache(cache_dir, fold_id)
        if self._cache_exists(out_path):
            logger.info("Fold %s 命中模型缓存", fold_id)
            return joblib.load(out_path)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        versions = self.data_module.fit_transform_versions(X_train, X_test)

        records, fold_preds = [], {}
        models_dir = cache_dir / "model_files" / f"fold_{fold_id:02d}"
        _ensure_dir(models_dir)

        # 构建任务列表
        tasks = []
        for fs_name, feats in feature_subsets.items():
            if not feats:
                continue
            for model_name in self.registry.models:
                tasks.append((fs_name, feats, model_name))

        def _run_combo(task):
            fs_name, feats, model_name = task
            try:
                ver = self.registry.data_req[model_name]
                Xtr = versions[ver]["X_train"][feats]
                Xte = versions[ver]["X_test"][feats]
                est, bp = self.modeling_module.optimize_and_fit(model_name, Xtr, y_train)
                pred = est.predict(Xte)
                key = f"{fs_name}__{model_name}"
                m = _metric_dict(y_test.values, pred)
                model_path = models_dir / f"{_safe_name(fs_name)}__{_safe_name(model_name)}.pkl"
                joblib.dump({
                    "estimator": est, "features": feats,
                    "feature_method": fs_name, "model_name": model_name,
                    "data_version": ver, "best_params": bp,
                }, model_path)
                logger.info(
                    "fold %s  %s__%s  R2=%.4f RMSE=%.4f",
                    fold_id, fs_name, model_name, m["R2"], m["RMSE"],
                )
                return {
                    "ok": True, "combo_key": key, "pred": pred,
                    "record": {
                        "fold_id": fold_id, "feature_method": fs_name,
                        "model_name": model_name, "combo_key": key,
                        "data_version": ver, "feature_count": len(feats),
                        "feature_list": str(feats),
                        "best_params": json.dumps(bp, ensure_ascii=False, default=str),
                        "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"],
                        "model_file": str(model_path),
                    },
                }
            except Exception as e:
                logger.exception("fold %s 组合失败: %s__%s", fold_id, fs_name, model_name)
                return {
                    "ok": False, "combo_key": f"{fs_name}__{model_name}", "pred": None,
                    "record": {
                        "fold_id": fold_id, "feature_method": fs_name,
                        "model_name": model_name,
                        "combo_key": f"{fs_name}__{model_name}",
                        "data_version": self.registry.data_req.get(model_name, "NA"),
                        "feature_count": len(feats), "feature_list": str(feats),
                        "best_params": "{}", "R2": np.nan, "RMSE": np.nan,
                        "MAE": np.nan, "model_file": "", "error": str(e),
                    },
                }

        # 串行或并行执行
        if n_jobs_inner <= 1 or len(tasks) <= 1:
            for t in tasks:
                r = _run_combo(t)
                records.append(r["record"])
                if r["ok"]:
                    fold_preds[r["combo_key"]] = r["pred"]
        else:
            logger.info("fold %s 内层并行: workers=%s tasks=%s", fold_id, n_jobs_inner, len(tasks))
            with ThreadPoolExecutor(max_workers=n_jobs_inner) as executor:
                futures = [executor.submit(_run_combo, t) for t in tasks]
                for f in as_completed(futures):
                    r = f.result()
                    records.append(r["record"])
                    if r["ok"]:
                        fold_preds[r["combo_key"]] = r["pred"]

        payload = {
            "fold_id": fold_id, "test_idx": test_idx,
            "records": records, "fold_preds": fold_preds,
        }
        _ensure_dir(out_path.parent)
        joblib.dump(payload, out_path)
        return payload

    # ══════════════════════════════════════════════════════
    # OOF 汇总
    # ══════════════════════════════════════════════════════

    def _aggregate_oof(
        self, cache_dir: Path, n_samples: int, fold_payloads: List[Dict],
    ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, List[str]]:
        storage = self._resolve_oof_storage()
        records = []

        if storage == "disk":
            all_keys = sorted({
                rec["combo_key"]
                for p in fold_payloads for rec in p["records"] if rec.get("combo_key")
            })
            key_to_col = {k: i for i, k in enumerate(all_keys)}
            chunks_dir = self._oof_chunks_dir(cache_dir)
            _ensure_dir(chunks_dir)
            for p in fold_payloads:
                records.extend(p["records"])
                fid = int(p["fold_id"])
                chunk_path = self._oof_chunk_path(cache_dir, fid)
                if self._cache_exists(chunk_path):
                    continue
                cache = joblib.load(self._fold_model_cache(cache_dir, fid))
                fp = cache.get("fold_preds", {})
                te = np.asarray(cache["test_idx"], dtype=int)
                mat = np.full((len(te), len(all_keys)), np.nan, dtype=np.float64)
                for k, pred in fp.items():
                    col = key_to_col.get(k)
                    if col is not None:
                        mat[:, col] = np.asarray(pred, dtype=np.float64)
                np.savez_compressed(chunk_path, fold_id=np.array([fid], dtype=np.int32), test_idx=te, preds=mat)
            meta_path = self._oof_meta_path(cache_dir)
            if not meta_path.exists():
                _save_json(meta_path, {
                    "storage": "disk", "n_samples": int(n_samples),
                    "all_keys": all_keys, "n_folds": len(fold_payloads),
                })
            return None, pd.DataFrame(records), all_keys

        # memory 模式
        all_keys = sorted({k for p in fold_payloads for k in p["fold_preds"]})
        oof = {k: np.full(n_samples, np.nan, dtype=float) for k in all_keys}
        for p in fold_payloads:
            te = p["test_idx"]
            records.extend(p["records"])
            for k, pred in p["fold_preds"].items():
                oof[k][te] = pred
        oof_df = pd.DataFrame({"row_index": np.arange(n_samples)})
        for k, v in oof.items():
            oof_df[k] = v
        return oof_df, pd.DataFrame(records), all_keys

    def _load_oof_matrix(self, cache_dir: Path, all_keys: List[str], selected_keys: Optional[List[str]] = None) -> np.ndarray:
        keys = selected_keys or all_keys
        meta = _load_json(self._oof_meta_path(cache_dir))
        n = int(meta["n_samples"])
        base_idx = {k: i for i, k in enumerate(all_keys)}
        sel_idx = [base_idx[k] for k in keys]
        mat = np.full((n, len(keys)), np.nan, dtype=np.float64)
        for f in sorted(self._oof_chunks_dir(cache_dir).glob("fold_*.npz")):
            with np.load(f) as d:
                te = d["test_idx"].astype(int)
                mat[np.ix_(te, np.arange(len(keys)))] = d["preds"][:, sel_idx]
        return mat

    def _compute_combo_metrics(
        self, y: pd.Series, combo_info: pd.DataFrame,
        cache_dir: Optional[Path] = None, oof_df: Optional[pd.DataFrame] = None,
        all_keys: Optional[List[str]] = None, chunk_size: int = 50,
    ) -> pd.DataFrame:
        rows = []
        yv = y.values
        lookup = combo_info.drop_duplicates("combo_key").set_index("combo_key")

        if oof_df is not None:
            iter_keys = [c for c in oof_df.columns if c != "row_index"]
            for c in iter_keys:
                p = oof_df[c].values
                mask = np.isfinite(p)
                if mask.sum() == 0 or float(mask.sum() / len(yv)) < 1.0 or c not in lookup.index:
                    continue
                m = _metric_dict(yv[mask], p[mask])
                s = lookup.loc[c]
                rows.append({
                    "combo_key": c, "feature_method": s["feature_method"],
                    "model_name": s["model_name"], "data_version": s["data_version"],
                    "feature_count": s["feature_count"],
                    "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"],
                })
        else:
            keys = list(all_keys)
            for i in range(0, len(keys), chunk_size):
                chunk = keys[i: i + chunk_size]
                cm = self._load_oof_matrix(cache_dir, all_keys=keys, selected_keys=chunk)
                for j, c in enumerate(chunk):
                    p = cm[:, j]
                    mask = np.isfinite(p)
                    if mask.sum() == 0 or float(mask.sum() / len(yv)) < 1.0 or c not in lookup.index:
                        continue
                    m = _metric_dict(yv[mask], p[mask])
                    s = lookup.loc[c]
                    rows.append({
                        "combo_key": c, "feature_method": s["feature_method"],
                        "model_name": s["model_name"], "data_version": s["data_version"],
                        "feature_count": s["feature_count"],
                        "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"],
                    })
                del cm
                gc.collect()
        return pd.DataFrame(rows).sort_values(["RMSE", "MAE"], ascending=True).reset_index(drop=True)

    # ══════════════════════════════════════════════════════
    # 全量重训 + 外部预测
    # ══════════════════════════════════════════════════════

    def _fit_final_submodels(
        self, models_dir: Path, X: pd.DataFrame, y: pd.Series,
        selected: pd.DataFrame, full_subsets: Dict[str, List[str]],
    ) -> Tuple[List[Dict], Dict]:
        versions = self.data_module.fit_versions_on_full(X)
        final = []
        out_dir = models_dir / "submodels"
        _ensure_dir(out_dir)
        for rank, row in selected.reset_index(drop=True).iterrows():
            fs = row["feature_method"]
            mn = row["model_name"]
            feats = full_subsets.get(fs, list(X.columns[:5]))
            if not feats:
                feats = list(X.columns[:5])
            ver = self.registry.data_req[mn]
            Xv = versions[ver]["X_full"][feats]
            est, bp = self.modeling_module.optimize_and_fit(mn, Xv, y)
            pred = est.predict(Xv)
            p = out_dir / f"{rank+1:02d}_{_safe_name(fs)}__{_safe_name(mn)}.pkl"
            joblib.dump({
                "estimator": est, "feature_method": fs, "model_name": mn,
                "data_version": ver, "features": feats, "best_params": bp,
            }, p)
            final.append({
                "rank": int(rank + 1), "combo_key": row["combo_key"],
                "feature_method": fs, "model_name": mn,
                "data_version": ver, "features": feats,
                "best_params": bp, "estimator": est,
                "fit_pred": pred, "model_file": str(p),
            })
        return final, versions

    def _predict_external(
        self, final_models: List[Dict], weights: np.ndarray, pred_file: str,
    ) -> pd.DataFrame:
        if not pred_file or not os.path.exists(pred_file):
            return pd.DataFrame()
            
        pred_df = DataModule._read_table(pred_file).copy()
        versions = self.data_module.transform_prediction_versions(pred_df)
        stack = []
        for m in final_models:
            feats = m["features"]
            Xp = versions[m["data_version"]].copy()
            for c in feats:
                if c not in Xp.columns:
                    Xp[c] = 0.0
            stack.append(m["estimator"].predict(Xp[feats]))
        final_pred = np.clip(np.column_stack(stack) @ weights, 0, None)

        id_col = None
        for cand in ("Pred_ID", "ID", "Station_ID", "StationID"):
            if cand in pred_df.columns:
                id_col = cand
                break
        ids = pred_df[id_col].values if id_col else np.arange(1, len(pred_df) + 1)
        return pd.DataFrame({"Pred_ID": ids, "Predicted_LUR": final_pred})

    # ══════════════════════════════════════════════════════
    # 公开入口 —— 单文件
    # ══════════════════════════════════════════════════════

    def run_single_file(
        self,
        train_file: str,
        predict_file: str,
        output_root: str,
    ) -> Dict[str, Any]:
        """处理单个数据集的完整流程"""
        train_path = Path(train_file)
        dataset_name = _safe_name(train_path.stem)
        
        # 二级目录分类存储
        cache_dir = Path(output_root) / "Cache" / dataset_name
        models_dir = Path(output_root) / "Models" / dataset_name
        figures_dir = Path(output_root) / "Figures" / dataset_name
        reports_dir = Path(output_root) / "Reports" / dataset_name
        preds_dir = Path(output_root) / "Predictions" / dataset_name
        
        _ensure_dir(cache_dir)
        _ensure_dir(models_dir)
        _ensure_dir(figures_dir)
        _ensure_dir(reports_dir)
        _ensure_dir(preds_dir)
        
        logger.info("开始处理: %s", dataset_name)

        # 1. 加载数据
        df, X, y, coords = self.data_module.load_training_data(train_file)
        outer_k = self.data_module.choose_outer_k(len(df))
        logger.info("样本=%s 特征=%s 外层CV=%s", len(df), X.shape[1], outer_k)

        # 2. 空间划分
        split_path = cache_dir / "data_split" / "outer_folds.pkl"
        if self._cache_exists(split_path):
            folds = joblib.load(split_path)
        else:
            folds = self.data_module.spatial_outer_split(coords, outer_k)
            _ensure_dir(split_path.parent)
            joblib.dump(folds, split_path)
        _save_json(cache_dir / "data_split" / "split_meta.json", {
            "dataset": dataset_name, "n_samples": int(len(df)),
            "n_features": int(X.shape[1]), "outer_k": outer_k,
            "target": self.data_module.target_col,
            "lon_col": self.data_module.lon_col,
            "lat_col": self.data_module.lat_col,
        })

        # 3. 外层 CV（串行或多进程并行）
        storage = self._resolve_oof_storage()
        n_jobs_fold = self._resolve_fold_workers(outer_k, X, y)
        n_jobs_inner = self._resolve_inner_workers(n_jobs_fold)
        logger.info(
            "并行策略: mode=%s fold_workers=%s inner_workers=%s",
            self._resolve_parallel_mode(), n_jobs_fold, n_jobs_inner,
        )

        fold_payloads = []
        if n_jobs_fold == 1:
            for fold_id, (tr, te) in enumerate(folds, start=1):
                logger.info("Fold %s/%s", fold_id, len(folds))
                subsets = self._run_feature_engineering_for_fold(cache_dir, fold_id, tr, te, X, y)
                fp = self._train_one_fold(cache_dir, fold_id, tr, te, X, y, subsets, n_jobs_inner)
                if storage == "disk":
                    fold_payloads.append({
                        "fold_id": fp["fold_id"], "test_idx": fp["test_idx"], "records": fp["records"],
                    })
                else:
                    fold_payloads.append(fp)
        else:
            fold_payloads = self._run_folds_parallel(
                cache_dir, folds, X, y, n_jobs_fold, n_jobs_inner, storage,
            )
        fold_payloads.sort(key=lambda p: p["fold_id"])

        # 4. OOF 汇总 + 指标
        oof_df, rec_df, all_keys = self._aggregate_oof(cache_dir, len(df), fold_payloads)
        rec_df.to_csv(reports_dir / "fold_model_records.csv", index=False)

        if storage == "disk":
            combo_df = self._compute_combo_metrics(y, rec_df, cache_dir=cache_dir, all_keys=all_keys)
        else:
            combo_df = self._compute_combo_metrics(y, rec_df, oof_df=oof_df)
            oof_df.to_csv(reports_dir / "oof_predictions_all_models.csv", index=False)
        combo_df.to_csv(reports_dir / "all_model_performance_oof.csv", index=False)

        if combo_df.empty:
            raise ValueError("无可用的完整 OOF 模型结果")

        # 5. Top N + Greedy 权重
        top = combo_df.head(self.config.top_ensemble_models).copy()
        top_keys = top["combo_key"].tolist()
        if not top_keys:
            raise ValueError("未筛选到可用于集成的模型")

        if storage == "disk":
            pred_matrix = self._load_oof_matrix(cache_dir, all_keys, top_keys)
        else:
            pred_matrix = oof_df[top_keys].values

        weights = self.ensemble_module.greedy_weights(pred_matrix, y.values)
        ensemble_pred = pred_matrix @ weights
        del pred_matrix
        gc.collect()

        ensemble_metrics = _metric_dict(y.values, ensemble_pred)
        logger.info(
            "OOF 集成: R2=%.4f RMSE=%.4f MAE=%.4f",
            ensemble_metrics["R2"], ensemble_metrics["RMSE"], ensemble_metrics["MAE"],
        )
        top["weight"] = weights
        top.to_csv(reports_dir / "ensemble_top_with_weights.csv", index=False)
        _save_json(reports_dir / "ensemble_metrics.json", ensemble_metrics)

        # 6. 全量重训
        full_feature_path = cache_dir / "feature_engineering" / "full_data_subsets.json"
        if self._cache_exists(full_feature_path):
            full_subsets = _load_json(full_feature_path)
        else:
            dummy = self.data_module.fit_transform_versions(X, X)
            full_subsets = self.feature_module.run_fold_feature_engineering(dummy, y)
            _save_json(full_feature_path, full_subsets)

        final_submodels, full_versions = self._fit_final_submodels(models_dir, X, y, top, full_subsets)
        fit_pred = np.column_stack([m["fit_pred"] for m in final_submodels]) @ weights
        fit_metrics = _metric_dict(y.values, fit_pred)
        logger.info(
            "全量重训: R2=%.4f RMSE=%.4f MAE=%.4f",
            fit_metrics["R2"], fit_metrics["RMSE"], fit_metrics["MAE"],
        )

        # 7. 外部预测
        ext_df = self._predict_external(final_submodels, weights, predict_file)
        pd.DataFrame({
            "Observed": y.values,
            "Model_Fitting_Pred": fit_pred,
            "Cross_Validation_Pred": ensemble_pred,
        }).to_csv(preds_dir / "train_fit_and_cv_predictions.csv", index=False)
        if not ext_df.empty:
            ext_df.to_csv(preds_dir / "external_predictions.csv", index=False)
        _save_json(reports_dir / "final_fit_metrics.json", fit_metrics)

        # 8. 保存最终模型
        model_payload = {
            "dataset_name": dataset_name,
            "target_col": self.data_module.target_col,
            "feature_cols": self.data_module.feature_cols,
            "weights": weights.tolist(),
            "ensemble_metrics": ensemble_metrics,
            "final_fit_metrics": fit_metrics,
            "submodels": [
                {
                    "rank": m["rank"], "combo_key": m["combo_key"],
                    "feature_method": m["feature_method"], "model_name": m["model_name"],
                    "data_version": m["data_version"], "features": m["features"],
                    "best_params": m["best_params"], "model_file": m["model_file"],
                }
                for m in final_submodels
            ],
            "scalers": self.data_module.scalers,
        }
        joblib.dump(model_payload, models_dir / "final_ensemble_model.pkl")

        # 9. 可视化（出版级）
        vis = VisualizationModule(figures_dir, config=self.config)
        scatter_path = vis.plot_scatter_residual(
            y_true=y.values, y_fit=fit_pred, y_cv=ensemble_pred,
            file_name="prediction_scatter_residual.png",
            dataset_name=dataset_name,
            fit_metrics=fit_metrics, cv_metrics=ensemble_metrics,
        )
        shap_path = vis.plot_shap_for_weighted_models(
            submodels=final_submodels, weights=weights,
            full_versions=full_versions,
            file_name="final_model_shap.png",
            max_samples=self.config.max_shap_samples,
            dataset_name=dataset_name,
        )
        
        # 10. 层级 SHAP 依赖图
        hierarchical = HierarchicalShapModule(figures_dir / "Hierarchical_SHAP", reports_dir, self.config)
        hierarchical.compute_and_plot(dataset_name, final_submodels, weights, X, full_versions)

        # 11. 汇总
        _save_json(reports_dir / "summary.json", {
            "dataset": dataset_name,
            "outer_k": outer_k,
            "n_samples": int(len(df)),
            "n_features": int(X.shape[1]),
            "ensemble_metrics": ensemble_metrics,
            "final_fit_metrics": fit_metrics,
            "scatter_figure": str(scatter_path),
            "shap_figure": str(shap_path),
            "predict_file": predict_file,
        })
        logger.info("数据集处理完成: %s", dataset_name)
        return {
            "dataset": dataset_name, "cache_dir": str(cache_dir),
            "ensemble_metrics": ensemble_metrics,
            "final_fit_metrics": fit_metrics,
            "ensemble_top": top,
            "external_predictions": ext_df,
            "submodels": model_payload["submodels"],
        }

    # ══════════════════════════════════════════════════════
    # 公开入口 —— 批量
    # ══════════════════════════════════════════════════════

    def run_batch(
        self,
        input_dir: str,
        predict_file: str,
        output_root: str,
    ) -> Dict[str, Any]:
        """批量处理目录下所有 xlsx 文件"""
        root = Path(input_dir)
        files = sorted(root.glob("*.xlsx"))
        if not files:
            raise ValueError(f"目录中未找到 xlsx 文件: {input_dir}")
        logger.info("批处理开始: 目录=%s 文件数=%s", input_dir, len(files))

        batch_results, all_ext, all_sub = [], [], []
        for f in files:
            logger.info("批处理中: %s", f.name)
            r = self.run_single_file(str(f), predict_file, output_root)
            batch_results.append({
                "dataset": r["dataset"],
                "ensemble_R2": r["ensemble_metrics"]["R2"],
                "ensemble_RMSE": r["ensemble_metrics"]["RMSE"],
                "ensemble_MAE": r["ensemble_metrics"]["MAE"],
                "fit_R2": r["final_fit_metrics"]["R2"],
                "fit_RMSE": r["final_fit_metrics"]["RMSE"],
                "fit_MAE": r["final_fit_metrics"]["MAE"],
            })
            if not r["external_predictions"].empty:
                ext = r["external_predictions"].rename(columns={"Predicted_LUR": r["dataset"]})
                all_ext.append(ext)
            for row in r["submodels"]:
                all_sub.append({
                    "dataset": r["dataset"], **{k: row[k] for k in row if k != "best_params"},
                    "best_params": json.dumps(row["best_params"], ensure_ascii=False, default=str),
                })

        out = Path(output_root) / "Reports"
        _ensure_dir(out)
        pd.DataFrame(batch_results).sort_values("dataset").to_excel(out / "BATCH_SUMMARY.xlsx", index=False)
        pd.DataFrame(all_sub).sort_values(["dataset", "rank"]).to_excel(out / "BATCH_ENSEMBLE_SUBMODELS.xlsx", index=False)
        merged = None
        if all_ext:
            for d in all_ext:
                merged = d if merged is None else pd.merge(merged, d, on="Pred_ID", how="outer")
            if merged is not None:
                merged.to_csv(Path(output_root) / "Predictions" / "BATCH_EXTERNAL_PREDICTIONS.csv", index=False)
        logger.info("批处理完成: %s", output_root)
        return {"n_files": len(files), "summary_file": str(out / "BATCH_SUMMARY.xlsx")}

    # ══════════════════════════════════════════════════════
    # 多进程并行执行外层 fold（含动态调度）
    # ══════════════════════════════════════════════════════

    def _run_folds_parallel(
        self, cache_dir, folds, X, y, n_jobs_fold, n_jobs_inner, storage,
    ) -> List[Dict]:
        cfg_dict = asdict(self.config)
        cfg_dict["n_jobs_fold"] = n_jobs_fold
        cfg_dict["n_jobs_inner"] = n_jobs_inner
        worker_config = PipelineConfig(**cfg_dict)

        fold_map = {fid: (tr, te) for fid, (tr, te) in enumerate(folds, 1)}
        args_list = [
            FoldWorkerArgs(fold_id=fid, train_idx=tr, test_idx=te, X=X, y=y, cache_dir=cache_dir, config=worker_config)
            for fid, (tr, te) in enumerate(folds, 1)
        ]

        fold_payloads = []
        target_inflight = n_jobs_fold
        retry_ids = []

        with ProcessPoolExecutor(max_workers=n_jobs_fold) as executor:
            pending = list(args_list)
            in_flight = {}

            while pending or in_flight:
                while pending and len(in_flight) < target_inflight:
                    a = pending.pop(0)
                    in_flight[executor.submit(_fold_worker, a)] = a.fold_id

                if not in_flight:
                    continue
                done, _ = wait(
                    list(in_flight.keys()),
                    timeout=max(0.5, self.config.scheduler_tick_sec),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    # 动态调度：监控内存
                    mem = self._get_memory_load_ratio()
                    if mem > 0.80 and target_inflight > 1:
                        target_inflight -= 1
                        logger.warning("内存偏高，降低并发至 %s", target_inflight)
                    elif 0 <= mem < 0.65 and target_inflight < n_jobs_fold:
                        target_inflight += 1
                    continue

                for future in done:
                    fid = in_flight.pop(future)
                    try:
                        fp = future.result()
                        if storage == "disk":
                            fold_payloads.append({
                                "fold_id": fp["fold_id"], "test_idx": fp["test_idx"], "records": fp["records"],
                            })
                        else:
                            fold_payloads.append(fp)
                    except Exception:
                        retry_ids.append(fid)
                        logger.exception("fold %s 子进程失败，稍后降级重试", fid)

                # 失败率过高时自动降并发
                total = len(fold_payloads) + len(retry_ids)
                if total > 0 and len(retry_ids) / total >= 0.3 and target_inflight > 1:
                    target_inflight = max(1, target_inflight - 1)

        # 降级重试：串行模式
        if retry_ids:
            logger.warning("降级重试 %s 个 fold", len(retry_ids))
            for fid in sorted(set(retry_ids)):
                tr, te = fold_map[fid]
                subsets = self._run_feature_engineering_for_fold(cache_dir, fid, tr, te, X, y)
                fp = self._train_one_fold(cache_dir, fid, tr, te, X, y, subsets, n_jobs_inner=1)
                if storage == "disk":
                    fold_payloads.append({
                        "fold_id": fp["fold_id"], "test_idx": fp["test_idx"], "records": fp["records"],
                    })
                else:
                    fold_payloads.append(fp)

        return fold_payloads
