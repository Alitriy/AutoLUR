"""
数据模块
负责数据加载、空间划分、标准化/归一化
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import PipelineConfig

logger = logging.getLogger("autolur")


def _norm_name(name: str) -> str:
    return str(name).replace("\u200b", "").replace("\ufeff", "").strip().lower()


class DataModule:
    """数据加载、空间划分、多版本缩放"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.lon_col = None
        self.lat_col = None
        self.target_col = None
        self.feature_cols: List[str] = []
        self.scalers: Dict[str, object] = {}

    # ── 加载 ──────────────────────────────────────────────

    def load_training_data(
        self, file_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
        df = self._read_table(file_path).copy()
        if df.shape[1] < 4:
            raise ValueError("训练数据列数不足，至少应包含目标列、经度、纬度和 1 个特征列")

        col_map = {_norm_name(c): c for c in df.columns}
        lon_col = self._find_col(col_map, ["longitude", "lon", "x"])
        lat_col = self._find_col(col_map, ["latitude", "lat", "y"])
        if lon_col is None or lat_col is None:
            raise ValueError("未检测到 Longitude / Latitude 列")

        target_col = df.columns[0]
        feature_cols = [c for c in df.columns if c not in (target_col, lon_col, lat_col)]
        if not feature_cols:
            raise ValueError("无可用于建模的特征列")

        self.lon_col, self.lat_col = lon_col, lat_col
        self.target_col = target_col
        self.feature_cols = feature_cols

        X = df[feature_cols].copy()
        y = df[target_col].copy()
        coords = df[[lon_col, lat_col]].copy()
        return df, X, y, coords

    # ── 空间划分 ──────────────────────────────────────────

    def choose_outer_k(self, n_samples: int) -> int:
        if n_samples <= self.config.small_n_threshold:
            return self.config.outer_cv_small_n
        return self.config.outer_cv_large_n

    def spatial_outer_split(
        self, coords: pd.DataFrame, k: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=20)
        groups = km.fit_predict(coords.values)
        splitter = GroupKFold(n_splits=k)
        return [
            (tr, te) for tr, te in splitter.split(coords.values, groups=groups)
        ]

    # ── 数据版本 ──────────────────────────────────────────

    def fit_transform_versions(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Dict[str, Dict]:
        versions = {
            "Raw": {"X_train": X_train.copy(), "X_test": X_test.copy(), "scaler": None}
        }
        std = StandardScaler()
        versions["Standardized"] = {
            "X_train": self._to_df(std.fit_transform(X_train), X_train),
            "X_test": self._to_df(std.transform(X_test), X_test),
            "scaler": std,
        }
        norm = MinMaxScaler()
        versions["Normalized"] = {
            "X_train": self._to_df(norm.fit_transform(X_train), X_train),
            "X_test": self._to_df(norm.transform(X_test), X_test),
            "scaler": norm,
        }
        return versions

    def fit_versions_on_full(
        self, X_full: pd.DataFrame
    ) -> Dict[str, Dict]:
        versions = {"Raw": {"X_full": X_full.copy(), "scaler": None}}
        std = StandardScaler()
        versions["Standardized"] = {
            "X_full": self._to_df(std.fit_transform(X_full), X_full),
            "scaler": std,
        }
        norm = MinMaxScaler()
        versions["Normalized"] = {
            "X_full": self._to_df(norm.fit_transform(X_full), X_full),
            "scaler": norm,
        }
        self.scalers = {"Standardized": std, "Normalized": norm}
        return versions

    def transform_prediction_versions(
        self, pred_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        raw = pred_df.copy()
        for c in self.feature_cols:
            if c not in raw.columns:
                raw[c] = 0.0
        raw = raw[self.feature_cols]
        data = {"Raw": raw}
        for name in ("Standardized", "Normalized"):
            if name in self.scalers:
                s = self.scalers[name]
                data[name] = pd.DataFrame(
                    s.transform(raw), columns=raw.columns, index=raw.index
                )
        return data

    # ── 工具 ──────────────────────────────────────────────

    @staticmethod
    def _read_table(file_path: str) -> pd.DataFrame:
        if file_path.lower().endswith(".csv"):
            return pd.read_csv(file_path)
        return pd.read_excel(file_path)

    @staticmethod
    def _find_col(col_map: dict, candidates: list):
        for c in candidates:
            if c in col_map:
                return col_map[c]
        return None

    @staticmethod
    def _to_df(arr, ref: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(arr, columns=ref.columns, index=ref.index)
