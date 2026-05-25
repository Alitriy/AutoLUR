"""
特征工程模块
11 种特征选择方法，每个 fold 独立执行
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

from .config import PipelineConfig

logger = logging.getLogger("autolur")


class FeatureModule:
    """11 种特征选择方法的统一入口"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ── 4.1 相关性 ──

    def corr_select(
        self, X: pd.DataFrame, y: pd.Series, method: str,
        threshold: float = 0.3, p_threshold: float = 0.05,
    ) -> List[str]:
        out = []
        for c in X.columns:
            v = X[c].corr(y, method=method)
            p = self._p_value(X[c], y, method)
            if np.isfinite(v) and np.isfinite(p) and abs(v) >= threshold and p < p_threshold:
                out.append(c)
        return out

    # ── 4.2 VIF ──

    def vif_select(
        self, X: pd.DataFrame, threshold: float = 5.0, min_features: int = 10,
    ) -> List[str]:
        cur = X.copy()
        cur = cur.loc[:, (cur != cur.iloc[0]).any()]
        min_keep = min(min_features, max(1, cur.shape[1]))
        for _ in range(80):
            if cur.shape[1] <= min_keep:
                break
            vals = []
            for i in range(cur.shape[1]):
                v = variance_inflation_factor(cur.values, i)
                vals.append(v if np.isfinite(v) else threshold + 1)
            idx = int(np.argmax(vals))
            if vals[idx] > threshold:
                cur = cur.drop(columns=[cur.columns[idx]])
            else:
                break
        return cur.columns.tolist()

    # ── 4.3 正则化 ──

    def lasso_select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        m = LassoCV(cv=self.config.inner_cv, random_state=self.config.random_state).fit(X, y)
        return X.columns[m.coef_ != 0].tolist()

    def enet_select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        m = ElasticNetCV(
            alphas=np.logspace(-4, 1, 50),
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=self.config.inner_cv,
            random_state=self.config.random_state,
        ).fit(X, y)
        return X.columns[m.coef_ != 0].tolist()

    # ── 4.4 模型重要性 ──

    def rf_importance_select(
        self, X: pd.DataFrame, y: pd.Series, cumulative: float = 0.85,
    ) -> List[str]:
        m = RandomForestRegressor(random_state=self.config.random_state, n_jobs=1)
        m.fit(X, y)
        s = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
        cs = s.cumsum()
        sel = s[cs <= cumulative].index.tolist()
        if not sel and len(s):
            sel = [s.index[0]]
        if len(sel) < len(s):
            nxt = len(sel)
            if nxt < len(s):
                sel.append(s.index[nxt])
        return sel

    def xgb_importance_select(
        self, X: pd.DataFrame, y: pd.Series, top_percentile: float = 0.85,
    ) -> List[str]:
        m = XGBRegressor(random_state=self.config.random_state, n_jobs=1)
        m.fit(X, y)
        d = m.get_booster().get_score(importance_type="gain")
        vals = pd.Series(
            [d.get(f, 0) for f in X.columns], index=X.columns,
        ).sort_values(ascending=False)
        n = max(1, int(len(vals) * top_percentile))
        return vals.head(n).index.tolist()

    def permutation_select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        m = RandomForestRegressor(random_state=self.config.random_state, n_jobs=1)
        m.fit(X, y)
        r = permutation_importance(
            m, X, y, n_repeats=5, random_state=self.config.random_state, n_jobs=1,
        )
        return X.columns[r.importances_mean > 0].tolist()

    # ── 4.5 RFECV ──

    def rfecv_select(self, X: pd.DataFrame, y: pd.Series, est_name: str) -> List[str]:
        if est_name == "Linear":
            est = LinearRegression()
        elif est_name == "Ridge":
            est = Ridge()
        else:
            est = SVR(kernel="linear")
        r = RFECV(estimator=est, step=1, cv=self.config.inner_cv, min_features_to_select=1)
        r.fit(X, y)
        return X.columns[r.support_].tolist()

    # ── 统一入口 ──

    def _limit(self, features: List[str], fallback: List[str]) -> List[str]:
        out = features[: self.config.max_features_per_subset]
        if not out:
            out = fallback[: min(5, len(fallback))]
        return out

    def run_fold_feature_engineering(
        self, versions: Dict[str, Dict], y_train: pd.Series,
    ) -> Dict[str, List[str]]:
        x_raw = versions["Raw"]["X_train"]
        x_std = versions["Standardized"]["X_train"]
        fallback = list(x_raw.columns)
        thr = self.config.corr_threshold

        subsets = {}
        subsets["Corr_Pearson"] = self._limit(self.corr_select(x_raw, y_train, "pearson", thr), fallback)
        subsets["Corr_Spearman"] = self._limit(self.corr_select(x_raw, y_train, "spearman", thr), fallback)
        subsets["VIF"] = self._limit(
            self.vif_select(x_raw, self.config.vif_threshold, self.config.vif_min_features), fallback,
        )
        subsets["Imp_RF"] = self._limit(self.rf_importance_select(x_raw, y_train), fallback)
        subsets["Imp_XGB"] = self._limit(self.xgb_importance_select(x_raw, y_train), fallback)
        subsets["Imp_Perm"] = self._limit(self.permutation_select(x_raw, y_train), fallback)
        subsets["Reg_Lasso"] = self._limit(self.lasso_select(x_std, y_train), fallback)
        subsets["Reg_ElasticNet"] = self._limit(self.enet_select(x_std, y_train), fallback)
        subsets["RFE_Linear"] = self._limit(self.rfecv_select(x_std, y_train, "Linear"), fallback)
        subsets["RFE_Ridge"] = self._limit(self.rfecv_select(x_std, y_train, "Ridge"), fallback)
        subsets["RFE_SVR"] = self._limit(self.rfecv_select(x_std, y_train, "SVR"), fallback)

        subsets = {k: v for k, v in subsets.items() if v}
        logger.info("特征工程完成，生成 %s 个子集", len(subsets))
        return subsets

    # ── 辅助 ──

    @staticmethod
    def _p_value(x: pd.Series, y: pd.Series, method: str) -> float:
        if method == "pearson":
            return pearsonr(x, y)[1]
        return spearmanr(x, y)[1]
