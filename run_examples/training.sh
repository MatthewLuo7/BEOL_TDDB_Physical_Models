# example commands for training the parameters of physical models
python train.py --pipeline-type GPR --training-path ./datasets/training_set/ --validation-path ./datasets/validation_set/ --batch-size 1 --num-workers 20 --n-trials 10
python train.py --pipeline-type DPM --model-type PowerLaw --training-path ./datasets/training_set/ --validation-path ./datasets/validation_set/ --batch-size 1 --num-workers 20 --n-trials 10
python train.py --pipeline-type DPM --model-type SqrtE --training-path ./datasets/training_set/ --validation-path ./datasets/validation_set/ --batch-size 1 --num-workers 20 --n-trials 10
python train.py --pipeline-type DPM --model-type InverseE --training-path ./datasets/training_set/ --validation-path ./datasets/validation_set/ --batch-size 1 --num-workers 20 --n-trials 10