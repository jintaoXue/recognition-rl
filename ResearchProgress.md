# recognition-rl

## 2022/12

### 1.排除horizon带来的问题：

horizon = 30 action_horizon = 10

```bash
(Pdb) state_.ego.shape
torch.Size([1, 30, 5])
(Pdb) state_.ego[...,-1]
tensor([[-0.966667, -0.933333, -0.900000, -0.866667, -0.833333, -0.800000, -0.766667, -0.733333, -0.700000, -0.666667, -0.633333, -0.600000, -0.566667, -0.533333, -0.500000, -0.466667, -0.433333, -0.400000, -0.366667, -0.333333, -0.300000, -0.266667, -0.233333, -0.200000, -0.166667, -0.133333, -0.100000, -0.066667, -0.033333, -0.000000]], device='cuda:0')

经过操作：(state_.ego = state_.ego[:,-horizon:,:])
(Pdb) state_.ego[...,-1]
tensor([[-0.300000, -0.266667, -0.233333, -0.200000, -0.166667, -0.133333, -0.100000, -0.066667, -0.033333, -0.000000]], device='cuda:0')

经过操作：（state_.ego[...,-1] = state_.ego[...,-1]*30/horizon）
(Pdb) state_.ego[...,-1]
tensor([[-0.900000, -0.800000, -0.700000, -0.600000, -0.500000, -0.400000, -0.300000, -0.200000, -0.100000, -0.000000]], device='cuda:0')
完成对齐
```

### 2.测试内存消耗，以及确定buffersize

目前监督学习的buffersize设置为130000

> 测试监督学习的内存消耗
>
> 1.测试horizon为30
>
> supervise buffersize  =  10000  horizon = 30  num_workers = 10->  43.2G cpu
>
> supervise buffersize  =  20000  horizon = 30 num_workers = 10 -> 44.4G cpu 
>
> supervise buffersize = 130000(max)  horizon = 30 num_workers = 10 -> 58.2G cpu 
>
> （结论：是horizon 为30的话10000消耗1.2G左右 ）
>
> 2.测试horizon为10
>
> supervise buffersize = 10000  horizon = 10 num_workers = 10 -> 43.0G cpu 
>
> supervise buffersize = 30000  horizon = 10 num_workers = 10 -> 45.2G cpu 
>
> supervise buffersize = 80000  horizon = 10 num_workers = 10 -> 50.4G cpu 
>
> （结论：是horizon 为30的话10000消耗1.05G左右 ）
>
> 3.测试一个worker加载训练的模型要多大
>
> supervise buffersize = 10000  horizon = 10 num_workers = 11 ->  cpu 

> 测试RL的内存消耗
>
> rl-recog buffersize  =  10000  horizon = 30  num_workers = 10->   cpu
>
> rl-recog buffersize  =  20000  horizon = 30 num_workers = 10 -> G cpu 
>
> rl-recog buffersize = 10000  horizon = 10 num_workers = 10 -> G cpu 
>
> rl-recog buffersize = 10000  horizon = 10 num_workers = 11 -> G cpu 

### 3.horizon的改变带来的对比

位置： ~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveSingleAgent

（1）**horizon为30：2022-11-26-12:32:44----Nothing--supervise-hrz30-act10**

![image-20221203162750274](imgs/2022-12-3-0.png)

验证精度：见MIner/11-29文件夹

（2）**horizon为10：2022-11-30-15:05:17----Nothing--supervise-hrz10-act10**

![](imgs/2022-12-03-2.png)

![](imgs/2022-12-03-3.png)

| 序号 | character_loss | recognition time |
| ---- | -------------- | ---------------- |
|      |                |                  |
|      |                |                  |
|      |                |                  |



### 4.Without attention性能对比

### 5.将历史轨迹进行降采样



### 6.下层action改为1

### 7.论文阅读继续