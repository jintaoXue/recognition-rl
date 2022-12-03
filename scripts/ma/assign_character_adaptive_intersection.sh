#!/bin/bash  

CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-2 --evaluate -d evaluate_ray_isac_adaptive_character_assign__intersection --model-index $2
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-2 --evaluate -d evaluate_ray_isac_adaptive_character_assign__intersection --model-index `expr $2 + 200`
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-2 --evaluate -d evaluate_ray_isac_adaptive_character_assign__intersection --model-index `expr $2 + 400`
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-2 --evaluate -d evaluate_ray_isac_adaptive_character_assign__intersection --model-index `expr $2 + 600`
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-2 --evaluate -d evaluate_ray_isac_adaptive_character_assign__intersection --model-index `expr $2 + 800`

