#!/bin/bash  

CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-5 --evaluate -d evaluate_ray_isac_robust_character_assign__bottleneck --model-index $2
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-6 --evaluate -d evaluate_ray_isac_robust_character_assign__intersection --model-index $2
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-7 --evaluate -d evaluate_ray_isac_robust_character_assign__merge --model-index $2
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-8 --evaluate -d evaluate_ray_isac_robust_character_assign__roundabout --model-index $2

