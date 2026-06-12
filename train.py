import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import numpy as np
from pathlib import Path
from typing import List
import json

from src.reliability_classifiers.binary_classifier import (
	optimize_binary_threshold,
	evaluate_binary_threshold,
	)
from src.multiprocessing_wrapper import batch_process_by_path
from src.param_optimizer import TwoStageWaferOptimizer
from src.csv_utils import (
	load_matrix_csv,
	matrix_to_sparse_points,
	)

def find_all_wafer_paths(base_path: str) -> List[Path]:
    """
    Find all wafer paths in the dataset matching the pattern: path/lot_xxx/csv/wafer_yyy
    
    遍历数据集目录，寻找所有符合层级规则的晶圆文件夹路径。
    匹配规则：根目录/lot_xxx/csv/wafer_yyy
    
    Args:
        base_path (str): The root directory of the dataset. 
                         数据集的根目录路径。
        
    Returns:
        List[str]: A sorted list of absolute or relative paths to the wafers. 
                   按字母顺序排序的满足条件的晶圆路径列表（字符串格式）。
    """
    root_dir = Path(base_path)
    
    # 确保根目录存在 / Ensure the base directory exists
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"The dataset path does not exist or is not a directory: {base_path}")

    wafer_paths = []
    
    # 使用 glob 进行严格的层级模式匹配 / Use glob for strict hierarchical pattern matching
    # "lot_*" matches lot_123, lot_ABC, etc.
    # "csv" is the exact folder name required.
    # "wafer_*" matches wafer_01, wafer_test, etc.
    search_pattern = "lot_*/csv/wafer_*"
    
    for path in root_dir.glob(search_pattern):
        # 通常 wafer_yyy 是一个包含数据的文件夹，做一次 is_dir() 断言可以过滤掉同名的意外文件
        # Ensure the matched path is actually a directory, not a file
        if path.is_dir():
            # 保留 Path 对象
            wafer_paths.append(path.resolve())
            
    # 返回排序后的列表，保证不同操作系统下读取的顺序一致，这对机器学习复现非常重要
    # Return sorted paths to guarantee deterministic behavior across different OS
    return sorted(wafer_paths)

def build_train_job_and_label(dataset_path):
	job_list = []
	labels = []
	for wafer_path in find_all_wafer_paths(dataset_path):
		job_list.append(
				{
					"vl_path": wafer_path / 'Space.csv',
					"ll_path": wafer_path / 'MS.csv',
					"mode": "ttf_only",
					"output_ttf_path": None,
					"output_binary_path": None
				}
			)
		class_val_matrix = load_matrix_csv(wafer_path / 'ExistenceClass.csv').astype(np.int32)
		_, _, class_val, _ = matrix_to_sparse_points(class_val_matrix)
		labels.append(np.where(class_val == 3, 1, 0))		# 1 or 2: bad chip (0); 3: good chip (1)

	return job_list, labels

def main():
	parser = argparse.ArgumentParser(description='Train a model')
	parser.add_argument('--pipeline-type', type=str, choices=['GPR', 'DPM'], help='Prediction Pipeline Type')
	parser.add_argument('--model-type', type=str, default=None, choices=['PowerLaw', 'SqrtE', 'InverseE'], help='Only for DPM pipelines')
	parser.add_argument('--training-path', type=str, help='Path for training set')
	parser.add_argument('--validation-path', type=str, help='Path for validation set')
	parser.add_argument('--fscore-beta', type=float, default=2.0, help='The beta parameter of F score')
	parser.add_argument('--n-trials', type=int, default=20, help='Training iterations')
	parser.add_argument('--batch-size', type=int, default=1, help='Batch size of wafers')
	parser.add_argument('--num-workers', type=int, default=4, help='Number of CPUs')
	parser.add_argument('--save-path', type=str, default=None, help='Output path for training results')
	args = parser.parse_args()
	if args.pipeline_type == 'GPR': args.model_type = 'GPR'
	print(args)

	# build job list and label
	train_jobs, train_labels = build_train_job_and_label(args.training_path)
	val_jobs, val_labels = build_train_job_and_label(args.validation_path)

	# build config
	config = {
	"pipeline_type": args.pipeline_type,
	"M_structures": 1_000_000,
	"F_target": 1e-4,
	"N_samples_per_dim": 8,
	"unit_model_kwargs":\
		{
			"model_type": args.model_type
		}
	}
	if args.pipeline_type == 'GPR':
		config["unit_model_kwargs"]["actual_vl_max"] = 40.0
		config["unit_model_kwargs"]["actual_ll_max"] = 20.0

	# optimization
	optimizer = TwoStageWaferOptimizer(
					train_jobs=train_jobs, 
					val_jobs=val_jobs, 
					train_labels=train_labels, 
					val_labels=val_labels,
					batch_process_fn=batch_process_by_path, 
					optimize_fn=optimize_binary_threshold, 
					evaluate_fn=evaluate_binary_threshold,
					base_config=config,
					batch_size=args.batch_size,
					num_workers=args.num_workers,
					beta=args.fscore_beta
					)

	optimized_config, optimization_metadata = optimizer.run_optimization(
														n_trials=args.n_trials
														)

	save_path = args.save_path if args.save_path else Path(f'./configs/{args.pipeline_type}_{args.model_type}_{args.fscore_beta:.1f}_{args.n_trials}/')
	if not save_path.exists():
		save_path.mkdir()

	with open(save_path / 'config.json', "w") as f:
		json.dump(optimized_config, f)

	with open(save_path / 'metadata.json', "w") as f:
		json.dump(optimization_metadata, f)

	optimizer.plot_history(save_image_path=save_path/'metric_curve.png')


if __name__ == "__main__":
	main()