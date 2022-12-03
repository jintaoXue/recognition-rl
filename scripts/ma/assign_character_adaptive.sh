#!/bin/bash  

CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-1 --evaluate -d evaluate_ray_isac_adaptive_character_assign__bottleneck --model-index $2
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-2 --evaluate -d evaluate_ray_isac_adaptive_character_assign__intersection --model-index $2
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-3 --evaluate -d evaluate_ray_isac_adaptive_character_assign__merge --model-index $2
CUDA_VISIBLE_DEVICES=$1 python run_ma_eval.py v-1-4 --evaluate -d evaluate_ray_isac_adaptive_character_assign__roundabout --model-index $2

