# recognition-rl


1.基于rl的性格辨识：
python run_sa.py v6-4-1
2.基于rl without attention
python run_sa.py v6-4-3
3.基于监督学习的性格辨识
python run_sa.py v6-5-6
4.监督学习 without attention
python run_sa.py v6-5-9











## evaluate

### multi-agent, social behavior

- 第一步：选择交互行为

```bash
### bottleneck
python run_ma_eval.py v-3-1 --evaluate -d evaluate_ray_isac_adaptive_character__social_behavior__bottleneck

### intersection
python run_ma_eval.py v-3-2 --evaluate -d evaluate_ray_isac_adaptive_character__social_behavior__intersection

### merge
python run_ma_eval.py v-3-3 --evaluate -d evaluate_ray_isac_adaptive_character__social_behavior__merge

### roundabout
python run_ma_eval.py v-3-4 --evaluate -d evaluate_ray_isac_adaptive_character__social_behavior__roundabout
```


注：上述命令，加入` --render --invert`可查看可视化，例如

```bash
python run_ma_eval.py v-3-1 --evaluate -d evaluate_ray_isac_adaptive_character__social_behavior__bottleneck --render --invert
```



- 第二步：可视化交互行为

```bash
### bottleneck
python eval_social_behavior.py v1

### intersection (todo)
python eval_social_behavior.py v2

### merge (todo)
python eval_social_behavior.py v3

### roundabout (todo)
python eval_social_behavior.py v4
```





