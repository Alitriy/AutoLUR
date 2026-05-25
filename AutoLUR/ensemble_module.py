"""
集成模块
Greedy 权重优化（最小化 RMSE）
"""
import numpy as np
from sklearn.metrics import mean_squared_error

from .config import PipelineConfig


class EnsembleModule:
    """基于 OOF 预测矩阵的贪心权重搜索"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def greedy_weights(
        self, pred_matrix: np.ndarray, y_true: np.ndarray,
    ) -> np.ndarray:
        """
        贪心搜索最优加权组合。

        每轮尝试给某个模型的计数 +1，选择使 RMSE 下降最多的模型。
        50 轮后归一化为权重。部分模型计数可能为 0（被自动剔除）。
        """
        n_models = pred_matrix.shape[1]
        counts = np.zeros(n_models, dtype=float)
        best_rmse = np.inf

        for _ in range(self.config.greedy_iterations):
            best_idx = None
            candidate_best = best_rmse

            for i in range(n_models):
                test = counts.copy()
                test[i] += 1.0
                w = test / test.sum()
                pred = pred_matrix @ w
                rmse = np.sqrt(mean_squared_error(y_true, pred))
                if rmse + self.config.greedy_min_improvement < candidate_best:
                    candidate_best = rmse
                    best_idx = i

            if best_idx is None:
                break
            counts[best_idx] += 1.0
            best_rmse = candidate_best

        if counts.sum() == 0:
            counts[:] = 1.0
        return counts / counts.sum()
