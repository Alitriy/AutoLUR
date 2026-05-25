"""
AutoLUR - 自动化 Land Use Regression 建模系统

使用空间交叉验证 + 多层特征工程 + 17 种模型 × 11 种特征选择
+ Greedy 集成 + 智能并行 + 出版级可视化 + 层级 SHAP 分析
"""
from .config import PipelineConfig
from .pipeline import LURPipeline
from .hierarchical_shap import HierarchicalShapModule

__all__ = ["PipelineConfig", "LURPipeline", "HierarchicalShapModule"]
__version__ = "2.0.0"
