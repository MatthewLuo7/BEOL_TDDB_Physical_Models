if __name__ == "__main__":
	from src.multiprocessing_wrapper import batch_process_by_path
	
	# 1. 加载并配置预测器
	# -------- gpr -------
	config = {
	"pipeline_type": "GPR",
	"M_structures": 1_000_000,
	"F_target": 1e-4,
	"N_samples_per_dim": 8,
	"unit_model_kwargs": {
	    "model_type": "GPR",
	    "actual_vl_max":40.0,
	    "actual_ll_max":20.0
	    }
	}

	# # ------- dpm -------
	# config = {
	# "pipeline_type": "DPM",
	# "M_structures": 1_000_000,
	# "F_target": 1e-4,
	# "N_samples_per_dim": 32,
	# "unit_model_kwargs": {
	#     "model_type": "PowerLaw"    # PowerLaw, SqrtE, InverseE
	#     }
	# }

	# 2. 构建纯路径任务列表
	# 不管文件在什么奇形怪状的路径下，只要把【输入】和【期望输出】配对好即可
	job_list = []
	for lot_idx in range(1, 4+1):
		for wafer_idx in range(1, 25+1):
			job_list.append(
					{
						"vl_path": f"../BEOL_TDDB_D2/data/lot_{lot_idx:03d}/csv/wafer_{wafer_idx:02d}/Space.csv",
						"ll_path": f"../BEOL_TDDB_D2/data/lot_{lot_idx:03d}/csv/wafer_{wafer_idx:02d}/MS.csv",
						"mode": "ttf_only",
						"output_ttf_path": None,
						"output_binary_path": None
					}
				)


	# 3. 轰油门直接运行
	results = batch_process_by_path(
		jobs_list=job_list,
		pipeline_config=config,
		batch_size=1,
		num_workers=8,
	)