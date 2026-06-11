from src.unit_level_models.weibull_gpr_model import load_weibull_gpr_model
from src.unit_level_models.dpm_models import PowerLawDPMModel, SqrtEDPMModel, InverseEDPMModel

if __name__ == "__main__":
	dpm_models = [PowerLawDPMModel(),
				  SqrtEDPMModel(),
				  InverseEDPMModel()]
	weibull_gpr_model = load_weibull_gpr_model()
	
	# 上层高阶工具（如 Wafer 仿真器）统一调用方式：
	for model in [weibull_gpr_model] + dpm_models:
		print(f"Loaded {model.__class__.__name__}")