#! /bin/bash
python3 run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_96_48 --model Transformer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1  >logs/LongForecasting/Transformer_Etth1_48.log
