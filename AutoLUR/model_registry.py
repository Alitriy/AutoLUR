"""
模型注册表
17 种回归模型 + 超参搜索空间
"""
from typing import Any, Dict

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    QuantileRegressor,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skopt.space import Categorical, Integer, Real
from xgboost import XGBRegressor


class ModelRegistry:
    """模型注册表：模型实例、所需数据版本、超参搜索空间"""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.data_req: Dict[str, str] = {}
        self.search_space: Dict[str, Dict] = {}
        self._register()

    def _reg(self, name: str, model: Any, data_type: str, space: Dict) -> None:
        self.models[name] = model
        self.data_req[name] = data_type
        self.search_space[name] = space

    def _register(self) -> None:
        # ── 5.1 Raw ──
        self._reg("OLS", LinearRegression(), "Raw", {})

        self._reg("DecisionTree", DecisionTreeRegressor(random_state=42), "Raw", {
            "max_depth": Integer(3, 30),
            "min_samples_leaf": Integer(2, 20),
        })

        self._reg("RandomForest", RandomForestRegressor(random_state=42, n_jobs=1), "Raw", {
            "n_estimators": Integer(100, 300),
            "max_depth": Integer(4, 14),
            "min_samples_leaf": Integer(2, 15),
            "min_samples_split": Integer(4, 20),
            "max_features": Categorical(["sqrt", "log2"]),
        })

        self._reg("ExtraTrees", ExtraTreesRegressor(random_state=42, n_jobs=1), "Raw", {
            "n_estimators": Integer(100, 300),
            "max_depth": Integer(4, 18),
            "min_samples_leaf": Integer(2, 10),
            "min_samples_split": Integer(4, 20),
            "max_features": Categorical(["sqrt", "log2"]),
        })

        self._reg("GradientBoosting", GradientBoostingRegressor(random_state=42), "Raw", {
            "n_estimators": Integer(100, 300),
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "max_depth": Integer(2, 6),
            "min_samples_leaf": Integer(2, 15),
            "subsample": Real(0.7, 1.0),
        })

        self._reg("XGBoost", XGBRegressor(random_state=42, n_jobs=1), "Raw", {
            "n_estimators": Integer(100, 350),
            "max_depth": Integer(3, 6),
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "subsample": Real(0.7, 1.0),
            "colsample_bytree": Real(0.7, 1.0),
            "reg_alpha": Real(0.01, 2.0),
            "reg_lambda": Real(0.1, 10.0),
        })

        self._reg("LightGBM", LGBMRegressor(random_state=42, n_jobs=1, verbose=-1), "Raw", {
            "num_leaves": Integer(15, 60),
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "feature_fraction": Real(0.7, 1.0),
            "bagging_fraction": Real(0.7, 1.0),
            "reg_alpha": Real(0.01, 3.0),
            "reg_lambda": Real(0.1, 10.0),
        })

        self._reg("CatBoost", CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False), "Raw", {
            "depth": Integer(3, 8),
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "iterations": Integer(100, 350),
            "l2_leaf_reg": Real(1.0, 12.0),
        })

        # ── 5.2 Standardized ──
        self._reg("Ridge", Ridge(), "Standardized", {
            "alpha": Real(1e-4, 1e2, prior="log-uniform"),
        })

        self._reg("LASSO", Lasso(), "Standardized", {
            "alpha": Real(1e-4, 1e1, prior="log-uniform"),
        })

        self._reg("ElasticNet", ElasticNet(), "Standardized", {
            "alpha": Real(1e-4, 1e1, prior="log-uniform"),
            "l1_ratio": Real(0.1, 0.9),
        })

        self._reg("BayesianRidge", BayesianRidge(), "Standardized", {})

        self._reg("Quantile", QuantileRegressor(quantile=0.5, solver="highs-ds"), "Standardized", {})

        self._reg("SVR", SVR(kernel="rbf"), "Standardized", {
            "C": Real(1e-1, 1e2, prior="log-uniform"),
            "gamma": Real(1e-4, 1e0, prior="log-uniform"),
            "epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
        })

        self._reg("MLP", MLPRegressor(random_state=42, max_iter=600), "Standardized", {
            "hidden_layer_sizes": Integer(30, 140),
            "alpha": Real(1e-5, 1e-1, prior="log-uniform"),
            "learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
        })

        # ── 5.3 Normalized ──
        self._reg("KNN", KNeighborsRegressor(), "Normalized", {
            "n_neighbors": Integer(3, 30),
            "weights": Categorical(["uniform", "distance"]),
        })

        # ── 5.4 PLSR ──
        self._reg("PLSR", PLSRegression(max_iter=1000), "Standardized", {
            "n_components": Integer(2, 20),
        })
