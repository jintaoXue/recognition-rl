

from builtins import breakpoint
import copy
import time

import os, glob
from os.path import join

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.distributions import Normal, MultivariateNormal

from rllib.utils import init_weights

import rllib
from rllib.basic import prefix
from rllib.template import MethodSingleAgent, Model
from rllib.template.model import FeatureExtractor, FeatureMapper

from utils.buffer import ReplayBufferMultiWorker
import tqdm
import numpy as np
class IndependentSACsupervise(MethodSingleAgent):
    dim_reward = 2
    
    gamma = 0.9

    reward_scale = 50
    target_entropy = None
    alpha_init = 1.0

    lr_critic = 5e-4
    lr_actor = 3e-4
    lr_tune = 0.5e-4

    tau = 0.005

    buffer_size = 800000
    batch_size = 128

    start_timesteps = 0
    # start_timesteps = 1  ## ! warning
    before_training_steps = 0

    save_model_interval = 5000
    
    sample_reuse = 24
    updated_iters = 0
    buffer_count = buffer_size
    buffer_len_prev = 0
    def __init__(self, config: rllib.basic.YamlConfig, writer):
        super().__init__(config, writer)

        self.actor = config.get('net_actor', Actor)(config).to(self.device)

        #todo
        self.actor.method_name = 'INDEPENDENTSAC_V0'
        # self.actor.model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
        # self.actor.model_num = 865800
        # self.actor.model_dir = '~/github/zdk/recognition-rl/models/origin_no_history_bottleneck/'
        # self.actor.model_num = 445600
        
        self.actor.model_dir = config.action_policy_model_dir
        self.actor.model_num = config.action_policy_model_num
        self.models_to_load = [self.actor]
        
        # [model.load_model() for model in self.models_to_load]
        [load_model(model) for model in self.models_to_load]
        self.actor.method_name = 'IndependentSACsupervise'
        #todo
        self.actor_target = copy.deepcopy(self.actor)
        self.models_to_save = [self.actor]
        
        for name, p in self.actor.named_parameters():
            if name.startswith('fe.global_head_recognition') or \
                name.startswith('fe.ego_embedding_recog') or \
                name.startswith('fe.recog_feature_mapper') or \
                name.startswith('fe.agent_embedding_recog'):
                p.requires_grad = True
            else : p.requires_grad = False

        self.actor_optimizer = Adam(filter(lambda x: x.requires_grad is not False ,self.actor.parameters()), lr=self.lr_actor)
        
        self.recog_loss = nn.MSELoss()#l1 loss
        # for name,param in self.actor.state_dict(keep_vars=True).items():
        #     print(name,param.requires_grad)     
        ### automatic entropy tuning

        # to adpat the child class method : IndependentSACsuperviseRoll which have initialized RolloutBuffer*

        self.buffer: ReplayBufferMultiWorker = config.get('buffer', rllib.buffer.ReplayBuffer)(config, self.buffer_size, self.batch_size, self.device)
    def update_parameters(self):
        if len(self.buffer) < self.start_timesteps + self.before_training_steps:
            return

        self.update_parameters_start()
        self.writer.add_scalar(f'{self.tag_name}/buffer_size', len(self.buffer), self.step_update)

        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state

        '''character MSE'''
        t1 = time.time()
        _,_,recog_character = self.actor(state)   
        t2 = time.time()
        real_character = state.obs_character[:,:,-1]
        recog_character = recog_character[~torch.isinf(real_character)]
        real_character = real_character[~torch.isinf(real_character)]
        
        error = torch.abs(recog_character - real_character)
        mean = error.mean()
        std = error.std()
        character_loss = self.recog_loss(recog_character, real_character)
        RMSE_loss = torch.sqrt(character_loss)
        self.actor_optimizer.zero_grad()
        RMSE_loss.backward()    
        self.actor_optimizer.step()

        self.writer.add_scalar(f'{self.tag_name}/loss_character',  RMSE_loss.detach().item(), self.step_update)
        self.writer.add_scalar(f'{self.tag_name}/mean_error',  mean.detach().item(), self.step_update)
        self.writer.add_scalar(f'{self.tag_name}/std',  std.detach().item(), self.step_update)
        self.writer.add_scalar(f'{self.tag_name}/recog_time', t2-t1, self.step_update)

        if self.step_update % self.save_model_interval == 0:
            self._save_model()

        return

    def update_parameters_(self, index, n_iters=1000):
        #setting 1  do not clear buffer 
        num_case = self.buffer.__len__() - self.buffer_len_prev
        self.buffer_len_prev = self.buffer.__len__()
        self.buffer_count -= num_case
        if self.buffer_count < 0 : 
            print('stop update') 
            return
        n_iters = int(num_case / self.batch_size) * self.sample_reuse 
        self.updated_iters += n_iters

        print('nume_case:{}, sample_reuse:{}, update iters:{}, updated iters:{}'.format(num_case, \
        self.sample_reuse, n_iters, self.updated_iters))

        # for i in tqdm.tqdm(range(n_iters)):
        #     self.update_parameters()
        for i in range(n_iters):
            self.update_parameters()
        #setting 2 clear buffer
        # num_case = self.buffer.__len__() 
        # self.buffer_count -= num_case
        # if self.buffer_count < 0 : 
        #     print('stop update') 
        #     return
        # n_iters = int(num_case / self.batch_size) * self.sample_reuse 
        # self.updated_iters += n_iters

        # print('buffer_len:{}, sample_reuse:{}, update iters:{}, updated iters:{}'.format(num_case, \
        # self.sample_reuse, n_iters, self.updated_iters))

        # # for i in tqdm.tqdm(range(n_iters)):
        # #     self.update_parameters()
        # for i in range(n_iters):
        #     self.update_parameters()
        # self.buffer.clear()

        return

    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()

        if self.step_select < self.start_timesteps:
            action = torch.Tensor(len(state), self.dim_action).uniform_(-1,1)
        else:
            # print('select: ', self.step_select)
            states = rllib.buffer.stack_data(state)
            self.buffer.pad_state(states)
            states = states.cat(dim=0)
            action, _, _ = self.actor.sample(states.to(self.device))
            action = action.cpu()
        return action

    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        if self.step_select < self.start_timesteps:
            action = torch.Tensor(1,self.dim_action).uniform_(-1,1)
        else:
            action, _, _ = self.actor.sample(state.to(self.device))
            action = action.cpu()
        return action
    
    def _update_model(self):
        # print('[update_parameters] soft update')
        rllib.utils.soft_update(self.actor_target, self.actor, self.tau)


class IndependentSACsuperviseRoll(IndependentSACsupervise):
    def __init__(self, config: rllib.basic.YamlConfig, writer):
        super().__init__(config, writer)
        self.buffer_size = 80000
        self.sample_reuse = 8
        self.batch_size = 128
        self.gamma = 0.99
        self.num_iters = int(self.buffer_size / self.batch_size) * self.sample_reuse
        self.buffer: rllib.buffer.RolloutBuffer = config.get('buffer', rllib.buffer.RolloutBuffer)(config, self.device, self.batch_size)
        self.save_model_interval = 500
    


    def update_parameters(self):
        if len(self.buffer) < self.buffer_size:
            return
        self.update_parameters_start()
        print(prefix(self) + 'update step: ', self.step_update)

        for _ in range(self.num_iters):
            self.step_train += 1
            '''load data batch'''   
            experience = self.buffer.sample(self.gamma)
            state = experience.state

            '''character MSE'''
            t1 = time.time()
            _,_,recog_character = self.actor(state)  
            t2 = time.time()
            real_character = state.obs_character[:,:,-1]
            recog_character = recog_character[~torch.isinf(real_character)]
            real_character = real_character[~torch.isinf(real_character)]
            
            # breakpoint()
    

    


class Actor(rllib.template.Model):
    logstd_min = -5
    logstd_max = 1

    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        self.mean_no = nn.Tanh()
        self.std_no = nn.Tanh()
        #todo
        self.fe = config.get('net_actor_fe', FeatureExtractor)(config, 0)
        self.mean = config.get('net_actor_fm', FeatureMapper)(config, 0, self.fe.dim_feature, config.dim_action)
        self.std = copy.deepcopy(self.mean)
        # self.recog = config.get('net_recog', RecognitionNet)(config).to(self.device)
        self.apply(init_weights)

        #evaluate
        # self.recog_loss = nn.MSELoss()

    def forward(self, state):        
        x = self.fe(state)

        mean = self.mean_no(self.mean(x))
        logstd = self.std_no(self.std(x))
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        return mean, logstd *0.5, self.fe.get_recog_obs_svos()


    def sample(self, state):
        mean, logstd, _ = self(state)

        cov = torch.diag_embed( torch.exp(logstd) )
        dist = MultivariateNormal(mean, cov)
        u = dist.rsample()


        # if mean.shape[0] == 1:
        #     print('    policy entropy: ', dist.entropy().detach().cpu())
        #     print('    policy mean:    ', mean.detach().cpu())
        #     print('    policy std:     ', torch.exp(logstd).detach().cpu())
        #     print()



        ### Enforcing Action Bound
        action = torch.tanh(u)
        logprob = dist.log_prob(u).unsqueeze(1) \
                - torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        return action, logprob, mean


    def sample_deprecated(self, state):
        mean, logstd = self(state)

        dist = Normal(mean, torch.exp(logstd))
        u = dist.rsample()

        ### Enforcing Action Bound
        action = torch.tanh(u)
        logprob = dist.log_prob(u) - torch.log(1-action.pow(2) + 1e-6)
        logprob = logprob.sum(dim=1, keepdim=True)

        return action, logprob, mean


def load_model(initial_model, model_num=None, model_dir=None):
    if model_dir == None:
        model_dir = initial_model.model_dir
    if model_num == None:
        model_num = initial_model.model_num

    model_dir = os.path.expanduser(model_dir)
    models_name = '_'.join([initial_model.method_name.upper(), initial_model.__class__.__name__, '*.pth'])
    file_paths = glob.glob(join(model_dir, models_name))
    file_names = [os.path.split(i)[-1] for i in file_paths]
    nums = [int(i.split('_')[-2]) for i in file_names]
    if model_num == -1:
        model_num = max(nums)

    print()
    print('[rllib.template.Model.load_model] model_dir: ', model_dir)
    print('[rllib.template.Model.load_model] models_name: ', models_name)
    print('[rllib.template.Model.load_model] file_paths length: ', len(file_paths))

    assert model_num in nums
    model_name = '_'.join([initial_model.method_name.upper(), initial_model.__class__.__name__, str(initial_model.model_id), str(model_num), '.pth'])
    model_path = join(model_dir, model_name)
    print('[rllib.template.Model.load_model] load model: ', model_path)
    initial_model.load_state_dict(torch.load(model_path), strict=False)

def write_character(file, character) :
    character = character.detach().cpu().numpy()
    # character = [item.detach().cpu().numpy() for item in character]
    # character = character.squeeze(2)
    #将numpy类型转化为list类型
    character=character.tolist()
    #将list转化为string类型
    str1=str(character)
    file.write(str1 + '\n\n')