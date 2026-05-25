# AutoLUR - 自动化土地利用回归 (Land Use Regression ,LUR) 建模系统

AutoLUR 是一个面向环境暴露分析与空间建模的自动化机器学习框架。它结合了空间交叉验证、多层特征工程、大规模模型集成以及创新的层级 SHAP 特征依赖图分析技术，旨在为研究人员提供端到端、一键式、出版级的 LUR 建模解决方案。

***

## 📖 目录

- [快速化新手教程](#-快速化新手教程)
  - [1. 环境准备](#1-环境准备)
  - [2. 数据放置](#2-数据放置)
  - [3. 启动程序](#3-启动程序)
- [⚙️ 项目核心原理](#-项目核心原理)
  - [集成建模逻辑](#1-集成建模逻辑)
  - [SHAP 特征重要性分析](#2-shap-特征重要性分析)
  - [层级 SHAP 依赖图分析](#3-层级-shap-依赖图分析-新增)
- [📂 项目架构与输出结构](#-项目架构与输出结构)

***

## 🚀 快速化新手教程

### 1. 环境准备

本项目需要 Python 3.9+ 环境。建议使用 Conda 或 venv 创建独立的虚拟环境。

```bash
# 1. 创建虚拟环境 (推荐使用 conda)
conda create -n autolur python=3.9 -y
conda activate autolur

# 2. 安装项目依赖
pip install -r requirements.txt
```

> **注意：** 项目深度依赖 `shap` 与 `shapiq` 库来实现层级的特征依赖性分析，请确保安装过程中未出现报错。

### 2. 数据放置

本项目已重构为高度自动化的目录结构。用户无需指定繁琐的路径参数，只需将数据放在正确的目录下即可：

1. 在项目根目录下，会自动生成一个名为 `Data` 的文件夹（如果没有，请手动创建）。
2. 将你需要训练的 `.xlsx` 或 `.csv` 格式的数据集直接放入 `Data` 文件夹中。
   - 数据格式要求：至少包含目标变量列（第1列）、经度列（`longitude`、`lon` 或 `x`）、纬度列（`latitude`、`lat` 或 `y`），以及若干特征列。

### 3. 启动程序

在命令行中进入项目根目录，直接运行：

```bash
python -m AutoLUR.run
```

程序会自动读取 `Data` 目录下的所有数据集并开始**批量建模**。运行产生的所有结果将会分类存放在项目根目录下的 `Results` 文件夹中。

如果你只希望针对单个文件建模，或需要进行外部预测：

```bash
# 单文件模式，并指定外部预测文件
python -m AutoLUR.run --train Data/NO2.xlsx --predict Data/predict.xlsx
```

***

## ⚙️ 项目核心原理

### 1. 集成建模逻辑

AutoLUR 的建模流程摒弃了单一模型的局限性，采用**多模型与多特征工程交叉组合**的策略：

- **11 种特征选择方法**：包括 Pearson/Spearman 相关性筛选、VIF 共线性过滤、Lasso/ElasticNet 正则化筛选、RF/XGBoost 树模型重要性筛选以及递归特征消除 (RFECV) 等。
- **17 种基础算法**：涵盖 OLS、Ridge、Lasso、ElasticNet、BayesianRidge、Quantile、SVR、KNN、DecisionTree、RandomForest、ExtraTrees、GradientBoosting、XGBoost、LightGBM、CatBoost、MLP、PLSRegression。
- **空间交叉验证**：使用 K-Means 聚类根据经纬度划分 Fold，进行严格的空间交叉验证 (Spatial CV)，避免空间自相关导致过拟合。
- **贪婪集成 (Greedy Ensemble)**：在所有“特征选择+算法”组合生成的验证集 (OOF) 预测结果中，提取表现最好的 Top N 模型，通过贪婪算法寻找最优加权组合，最小化 RMSE。

### 2. SHAP 特征重要性分析

在最终的集成模型中，系统会根据每个子模型的权重，自动分配并计算对应的 SHAP 值：

- 对于树模型（如 XGBoost、RandomForest），采用高效的 `TreeExplainer`。
- 对于线性模型，采用 `LinearExplainer`。
- 其他模型使用 `KernelExplainer`。
  最终融合生成具有**出版级质量**的双轴 SHAP 特征重要性图（蜂巢图 + 带有百分比标注的半透明条形图）。

### 3. 层级 SHAP 依赖图分析

为了深度挖掘特征与特征之间的复杂交互作用，AutoLUR 创新性地引入了 **层级 SHAP **解析架构。传统的交互作用计算受限于算法支持度，而我们的分层架构保证了 100% 的子模型覆盖率：

- **Layer 1 (Exact - TreeExplainer)**：对于原生支持交互值计算的树模型，采用精确的 `shap.TreeExplainer` 计算特征间的精确交互值。
- **Layer 2 (Conditional Exact - Linear/PLS)**：对于线性模型与偏最小二乘回归，我们通过仿射空间映射（基于经验均值背景）精确重建特征的独立贡献，提取伪交互。
- **Layer 3 (Approximate - KernelSHAP-IQ)**：对于不支持直接交互计算的黑盒模型（如 SVR、KNN、Quantile），系统自动降级调用前沿的 `shapiq` (TabularExplainer + k-SII 索引) 进行高阶交互效应近似估算。

该模块会自动寻找与核心特征交互作用最强的“伙伴特征 (Partner Feature)”，并输出带有颜色映射的特征依赖散点图（包含单图与拼接面板图）。

***

## 📂 项目架构与输出结构

### 代码模块结构

```text
AutoLUR/
├── __init__.py           # 包导出与版本定义
├── config.py             # PipelineConfig：所有核心参数的统一数据类
├── run.py                # 命令行入口，解析参数并调用 Pipeline
├── pipeline.py           # 核心编排：数据流转、外层CV、并行调度、集成、全量重训
├── data_module.py        # 数据加载、坐标识别、空间划分(KMeans)、特征多版本标准化
├── feature_module.py     # 11 种特征工程选择算法的具体实现
├── model_registry.py     # 17 种回归算法实例定义及 BayesSearchCV 超参搜索空间
├── modeling_module.py    # 针对特定特征和数据版本的模型超参寻优与训练
├── ensemble_module.py    # 贪心搜索：基于 OOF 矩阵优化子模型权重
├── visualization.py      # 出版级可视化：残差散点图、加权 SHAP 重要性图
└── hierarchical_shap.py  # 层级 SHAP 分析：Tree/Linear/shapiq 交互效应计算与依赖图绘制
```

### 规范化输出结构 (`Results/`)

每次运行结束后，所有的产物都会被归档至 `Results/` 下对应的二级目录中：

```text
Results/
├── Models/              # 存放所有模型文件 (pkl)
│   └── DatasetName/     # 包含最终的 final_ensemble_model.pkl 及所有重训子模型
├── Figures/             # 存放出版级图片
│   └── DatasetName/
│       ├── prediction_scatter_residual.png    # 残差与拟合分布图
│       ├── final_model_shap.png               # SHAP 综合重要性图
│       └── Hierarchical_SHAP/                 # 新增的特征依赖散点图(单图与面板图)
├── Reports/             # 存放所有的分析报表与评估结果 (csv/json/xlsx)
│   ├── DatasetName/
│   │   ├── fold_model_records.csv             # 各个Fold的模型表现
│   │   ├── ensemble_top_with_weights.csv      # TopN 组合及其权重
│   │   └── ... (各种层级 SHAP 分析的审计日志与统计结果)
│   └── BATCH_SUMMARY.xlsx                     # 批量运行的整体汇总报告
├── Predictions/         # 存放对外部数据的预测结果
│   ├── DatasetName/
│   └── BATCH_EXTERNAL_PREDICTIONS.csv
└── Cache/               # 运行时缓存 (可安全删除以释放空间)
```

