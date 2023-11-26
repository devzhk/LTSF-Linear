python3 run_longExp.py --is_training 1 --model_id sub1-biking --model Transformer --learning_rate 0.00002 # train
python3 run_longExp.py --is_training 0 --model_id sub1-biking --model Transformer --do_predict 

CUDA_VISIBLE_DEVICES=0 python3 run_longExp.py --model_id allTrain --model Transformer --learning_rate 0.0001 --d_layers 2
CUDA_VISIBLE_DEVICES=0 python3 run_longExp.py --model_id allTrain --model Transformer --learning_rate 0.0001 --d_layers 2 --is_training 0 --do_predict --data_path Subject_5-cleaned-VR.csv