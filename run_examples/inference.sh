# example commands for run physical models
python inference.py --config-path ./configs/GPR_GPR_2.0_10/config.json 		--test-path ./datasets/test_set/ --save-path ./output/test_set/ --num-workers 20
python inference.py --config-path ./configs/DPM_PowerLaw_2.0_10/config.json --test-path ./datasets/test_set/ --save-path ./output/test_set/ --num-workers 20
python inference.py --config-path ./configs/DPM_SqrtE_2.0_10/config.json 	--test-path ./datasets/test_set/ --save-path ./output/test_set/ --num-workers 20
python inference.py --config-path ./configs/DPM_InverseE_2.0_10/config.json --test-path ./datasets/test_set/ --save-path ./output/test_set/ --num-workers 20