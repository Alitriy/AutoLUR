"""
建模模块
BayesSearchCV 超参搜索 + 模型训练
"""
import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.base import clone
from skopt import BayesSearchCV
from skopt.space import Integer

from .config import PipelineConfig
from .model_registry import ModelRegistry

logger = logging.getLogger("autolur")


class ModelingModule:
    """对单个 (特征子集 × 模型) 组合做超参搜索并训练"""

    def __init__(self, config: PipelineConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry

    def optimize_and_fit(
        self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
    ) -> Tuple[Any, Dict[str, Any]]:
        estimator = clone(self.registry.models[model_name])
        space = dict(self.registry.search_space[model_name])

        # PLSR 组件数上限受限于特征数和样本数
        if model_name == "PLSR":
            ub = max(1, min(20, X_train.shape[1], len(X_train) - 1))
            space = {"n_components": Integer(1, ub)}

        if not space:
            estimator.fit(X_train, y_train)
            return estimator, {}

        opt = BayesSearchCV(
            estimator=estimator,
            search_spaces=space,
            n_iter=self.config.bayes_iter,
            cv=self.config.inner_cv,
            n_jobs=1,
            scoring="neg_root_mean_squared_error",
            random_state=self.config.random_state,
            verbose=0,
        )
        try:
            opt.fit(X_train, y_train)
            best = opt.best_estimator_
            best.fit(X_train, y_train)
            return best, opt.best_params_
        except Exception:
            logger.exception("BayesSearchCV 失败，回退默认参数。model=%s", model_name)
            estimator.fit(X_train, y_train)
            return estimator, {}
