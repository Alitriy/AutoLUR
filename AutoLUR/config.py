"""
AutoLUR 配置模块
使用 dataclass 定义所有可调参数
"""
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # ── 随机种子 ──
    random_state: int = 42

    # ── 交叉验证 ──
    inner_cv: int = 3              # 内层 BayesSearchCV 折数
    bayes_iter: int = 20           # BayesSearchCV 迭代次数
    outer_cv_small_n: int = 5      # 样本 ≤ small_n_threshold 时外层折数
    outer_cv_large_n: int = 10     # 样本 > small_n_threshold 时外层折数
    small_n_threshold: int = 1000  # 小样本/大样本分界

    # ── 特征工程 ──
    max_features_per_subset: int = 20  # 每种方法最多保留特征数
    corr_threshold: float = 0.3        # 相关性筛选阈值
    vif_threshold: float = 5.0         # VIF 阈值
    vif_min_features: int = 15         # VIF 最少保留特征数

    # ── 集成 ──
    top_ensemble_models: int = 10      # 参与集成的候选模型数
    greedy_iterations: int = 50        # Greedy 权重迭代轮数
    greedy_min_improvement: float = 1e-6

    # ── 缓存 ──
    use_cache: bool = True

    # ── OOF 存储 ──
    oof_storage: str = "memory"        # "memory" 或 "disk"

    # ── 并行策略 ──
    n_jobs_fold: int = 0               # 0=自动, 1=串行, >1=指定
    n_jobs_inner: int = 0
    parallel_mode: str = "auto"        # off / conservative / balanced / aggressive / auto
    cpu_reserve_cores: int = 2
    mem_util_target: float = 0.75
    max_workers_cap: int = 16
    scheduler_tick_sec: float = 5.0

    # ── 可视化 ──
    max_shap_samples: int = 300
    shap_max_display: int = 10
    fig_dpi: int = 300
    fig_font_family: str = "Times New Roman"
    fig_font_size: int = 22
    scatter_figsize: tuple = (10, 9.6)
    shap_figsize: tuple = (10.5, 4.8)
