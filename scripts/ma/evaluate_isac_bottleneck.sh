

for fp in $(seq 0.0 .1 1.0)
do
echo 'start evaluate svo:'$fp; 
CUDA_VISIBLE_DEVICES=$0 python run_ma_eval.py vtrue-1 --evaluate --svo $fp 

done