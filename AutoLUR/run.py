#!/usr/bin/env python
"""
AutoLUR 命令行入口

用法:
  # 单文件模式 (指定具体文件)
  python -m AutoLUR.run --train Data/NO2.xlsx --predict Data/predict.xlsx

  # 批量模式 (默认读取 Data 目录下的所有 xlsx)
  python -m AutoLUR.run --batch Data/ --predict Data/predict.xlsx

  # 自动化模式 (只需在 Data/ 放置数据)
  python -m AutoLUR.run
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from AutoLUR import LURPipeline, PipelineConfig

def _setup_logger(level: str = "INFO"):
    logger = logging.getLogger("autolur")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def main():
    parser = argparse.ArgumentParser(
        description="AutoLUR - 自动化 Land Use Regression 建模系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 输入输出
    group_io = parser.add_argument_group("输入输出")
    group_io.add_argument("--train", help="单个训练数据文件 (.xlsx/.csv)")
    group_io.add_argument("--batch", default="Data", help="批量模式：包含多个 xlsx 的目录 (默认 Data/)")
    group_io.add_argument("--predict", help="预测数据文件 (可选)")
    group_io.add_argument("--output", default="Results", help="输出根目录 (默认 Results/)")

    # 模型配置
    group_model = parser.add_argument_group("模型配置")
    group_model.add_argument("--top_n", type=int, default=10, help="参与集成的 Top N 模型 (默认 10)")
    group_model.add_argument("--bayes_iter", type=int, default=20, help="BayesSearchCV 迭代数 (默认 20)")
    group_model.add_argument("--inner_cv", type=int, default=3, help="内层 CV 折数 (默认 3)")
    group_model.add_argument("--corr_threshold", type=float, default=0.3, help="相关性阈值 (默认 0.3)")

    # 并行策略
    group_par = parser.add_argument_group("并行策略")
    group_par.add_argument(
        "--parallel", default="auto",
        choices=["off", "conservative", "balanced", "aggressive", "auto"],
        help="并行模式 (默认 auto)",
    )
    group_par.add_argument("--fold_workers", type=int, default=0, help="外层 fold 并行数 (0=自动)")
    group_par.add_argument("--oof_storage", default="memory", choices=["memory", "disk"],
                           help="OOF 存储方式 (默认 memory)")

    # 可视化
    group_vis = parser.add_argument_group("可视化")
    group_vis.add_argument("--shap_max_display", type=int, default=10, help="SHAP 显示特征数 (默认 10)")
    group_vis.add_argument("--dpi", type=int, default=300, help="图片 DPI (默认 300)")

    # 其他
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--no_cache", action="store_true", help="禁用缓存")

    args = parser.parse_args()
    _setup_logger(args.log_level)

    # 自动创建 Data 和 Results 目录
    os.makedirs("Data", exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    if not args.train:
        # 检查 Data 目录下是否有 xlsx
        data_dir = Path(args.batch)
        if not data_dir.exists() or not list(data_dir.glob("*.xlsx")):
            print(f"============================================================")
            print(f"请将用于建模的数据集(.xlsx)放入 '{args.batch}' 目录中后重试。")
            print(f"============================================================")
            return

    config = PipelineConfig(
        top_ensemble_models=args.top_n,
        bayes_iter=args.bayes_iter,
        inner_cv=args.inner_cv,
        corr_threshold=args.corr_threshold,
        parallel_mode=args.parallel,
        n_jobs_fold=args.fold_workers,
        oof_storage=args.oof_storage,
        shap_max_display=args.shap_max_display,
        fig_dpi=args.dpi,
        use_cache=not args.no_cache,
    )

    pipeline = LURPipeline(config=config)

    if args.train:
        result = pipeline.run_single_file(args.train, args.predict, args.output)
        print(f"\n{'='*60}")
        print(f"完成: {result['dataset']}")
        print(f"OOF 集成 R²: {result['ensemble_metrics']['R2']:.4f}")
        print(f"全量拟合 R²: {result['final_fit_metrics']['R2']:.4f}")
        print(f"输出目录: {args.output}")
        print(f"{'='*60}")
    else:
        result = pipeline.run_batch(args.batch, args.predict, args.output)
        print(f"\n{'='*60}")
        print(f"批处理完成: {result['n_files']} 个数据集")
        print(f"汇总文件: {result['summary_file']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
