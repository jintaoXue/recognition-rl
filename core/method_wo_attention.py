

from builtins import breakpoint
import copy
from turtle import update
from importlib_metadata import re
# from markupsafe import string
import numpy as np


import os, glob
from os.path import join
import torch
import torch.nn as nn
from torch.optim import Adam

from torch.distributions import Normal, MultivariateNormal

from rllib.utils import init_weights

import rllib
from rllib.template import MethodSingleAgent, Model
from rllib.template.model import FeatureExtractor, FeatureMapper

from core.recognition_net import RecognitionWoAttention


class IndependentSAC_woatt(MethodSingleAgent):
    dim_reward = 2
    
    gamma = 0.9

    reward_scale = 50
    target_entropy = None
    alpha_init = 1.0

    lr_critic = 5e-4
    lr_actor = 3e-4
    lr_tune = 1e-4

    tau = 0.005

    buffer_size = 130000
    batch_size = 128

    # start_timesteps = 30000
    start_timesteps = 30000
    # start_timesteps = 1000  ## ! warning
    before_training_steps = 0

    save_model_interval = 200


    def __init__(self, config: rllib.basic.YamlConfig, writer):
        super().__init__(config, writer)

        self.actor = config.get('net_actor', Actor)(config).to(self.device)


        #todo
        self.actor.method_name = 'INDEPENDENTSAC_V0'
        # self.actor.model_dir =     '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
        # self.actor.model_num = 865800
        self.actor.model_dir = '~/github/zdk/recognition-rl/models/origin_no_history_bottleneck/'
        self.actor.model_num = 445600
        self.models_to_load = [self.actor]
        # [model.load_model() for model in self.models_to_load]
        [load_model(model) for model in self.models_to_load]
        self.actor.method_name = 'IndependentSAC_supervise'
        #todo
        self.actor_target = copy.deepcopy(self.actor)
        self.models_to_save = [self.actor]

        self.recog_optimizer= Adam(self.actor.recog.parameters(), lr=self.lr_actor)
        
        self.recog_loss = nn.MSELoss()#l1 loss
        # for name,param in self.actor.state_dict(keep_vars=True).items():
        #     print(name,param.requires_grad)     
        ### automatic entropy tuning

        self.buffer: rllib.buffer.ReplayBuffer = config.get('buffer', rllib.buffer.ReplayBuffer)(config, self.buffer_size, self.batch_size, self.device)

    def update_parameters(self):
        if len(self.buffer) < self.start_timesteps + self.before_training_steps:
            return
        self.update_parameters_start()
        self.writer.add_scalar(f'{self.tag_name}/buffer_size', len(self.buffer), self.step_update)

        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward
        done = experience.done

        '''character MSE'''

        recog_character = self.actor.recog(state)    
        real_character = state.obs_character[:,:,-1]
        recog_character = recog_character[~torch.isinf(real_character)]
        real_character = real_character[~torch.isinf(real_character)]
        
        # breakpoint()
        # real_character = torch.where(real_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), real_character)
        # recog_character = torch.where(recog_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), recog_character)
        character_loss = self.recog_loss(recog_character, real_character)
        RMSE_loss = torch.sqrt(character_loss)
        self.recog_optimizer.zero_grad()
        # character_loss.backward()
        RMSE_loss.backward()    
        self.recog_optimizer.step()
        file = open(self.output_dir + '/' + 'character.txt', 'w')
        write_character(file, recog_character)
        write_character(file, real_character)
        write_character(file, recog_character - real_character)
        file.write('*******************************\n')
        file.close()
        self.writer.add_scalar(f'{self.tag_name}/loss_character', character_loss.detach().item(), self.step_update)   

        # self.writer.add_scalar(f'{self.tag_name}/alpha', self.alpha.detach().item(), self.step_update)

        self._update_model()
        if self.step_update % self.save_model_interval == 0:
            self._save_model()

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

class IndependentSAC_recog_woattn(MethodSingleAgent):
    dim_reward = 2
    
    gamma = 0.9

    reward_scale = 50
    target_entropy = None
    alpha_init = 1.0

    lr_critic = 5e-4
    lr_actor = 1e-4
    lr_tune = 1e-4

    tau = 0.005

    buffer_size = 750000
    batch_size = 128

    start_timesteps = 0
    # start_timesteps = 1000  ## ! warning
    before_training_steps = 0

    save_model_interval = 1000


    def __init__(self, config: rllib.basic.YamlConfig, writer):
        super().__init__(config, writer)

        self.critic = config.get('net_critic', Critic)(config).to(self.device)
        self.actor = config.get('net_actor', Actor)(config).to(self.device)
        #todo
        self.actor.method_name = 'INDEPENDENTSAC_V0'
        # self.actor.model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
        # self.actor.model_num = 865800
        self.actor.model_dir = '~/github/zdk/recognition-rl/models/origin_no_history_bottleneck/'
        self.actor.model_num = 445600

        # self.critic.method_name = 'INDEPENDENTSAC_V0'
        # self.critic.model_dir = '~/github/zdk/recognition-rl/models/origin_no_history_bottleneck/'
        # self.critic.model_num = 445600
        # self.critic.model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
        # self.critic.model_num = 865800
        self.models_to_load = [self.actor]

        # [model.load_model() for model in self.models_to_load]
        [load_model(model) for model in self.models_to_load]
        self.actor.method_name = 'IndependentSAC_recog'
        self.critic.method_name = 'IndependentSAC_recog'
        for name, p in self.actor.named_parameters():
            if name.startswith('fe'): p.requires_grad = False
            if name.startswith('mean'): p.requires_grad = False
            if name.startswith('std'): p.requires_grad = False
        # for name, p in self.critic.named_parameters():
        #     if name.startswith('fe'): p.requires_grad = False
        #     if name.startswith('m1'): p.requires_grad = False
        #     if name.startswith('m2'): p.requires_grad = False
        # self.critic.method_name = 'IndependentSAC_recog'
        #todo
        self.critic_target = copy.deepcopy(self.critic)
        self.models_to_save = [self.actor, self.critic]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = Adam(filter(lambda x: x.requires_grad is not False ,self.actor.parameters()), lr=self.lr_actor)
        
        # self.recog_optimizer= Adam(self.actor.recog.parameters(), lr=self.lr_actor)
        self.critic_loss = nn.MSELoss()
        self.character_loss = nn.MSELoss()
        # for name,param in self.actor.state_dict(keep_vars=True).items():
        #     print(name,param.requires_grad)   
          
        ### automatic entropy tuning
        if self.target_entropy == None:
            self.target_entropy = -np.prod((self.dim_action,)).item()
        self.log_alpha = torch.full((), np.log(self.alpha_init), requires_grad=True, dtype=self.dtype, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_tune)

        self.buffer: rllib.buffer.ReplayBuffer = config.get('buffer', rllib.buffer.ReplayBuffer)(config, self.buffer_size, self.batch_size, self.device)

    def update_parameters(self):
        if len(self.buffer) < self.start_timesteps + self.before_training_steps:
            return
        self.update_parameters_start()
        self.writer.add_scalar(f'{self.tag_name}/buffer_size', len(self.buffer), self.step_update)

        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward
        done = experience.done

        '''critic'''
        with torch.no_grad():
            next_action, next_logprob, _ = self.actor.sample(next_state)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_logprob
            target_q = reward + self.gamma * (1-done) * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = (self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''actor'''
        action, logprob, _ = self.actor.sample(state)
        actor_loss = (-self.critic.q1(state, action) + self.alpha * logprob).mean() 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # for name, p in self.actor.recog.named_parameters():
        #     print(name, p)  
        # time.sleep(2)
        '''automatic entropy tuning'''
        alpha_loss = self.log_alpha.exp() * (-logprob.mean() - self.target_entropy).detach()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()
        
        '''character MSE'''
        recog_charater = self.actor.recog(state)    
        real_character = state.obs_character[:,:,-1]
        
        recog_charater = torch.where(real_character == np.inf, torch.tensor(np.inf, dtype=torch.float32, device=state.obs.device),recog_charater)
        real_character = real_character[~torch.isinf(real_character)]
        recog_charater = recog_charater[~torch.isinf(recog_charater)]
        # breakpoint()
        # real_character = torch.where(real_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), real_character)
        # recog_charater = torch.where(recog_charater == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), recog_charater)
        character_loss = self.character_loss(recog_charater, real_character)
        RMSE_loss = torch.sqrt(character_loss)
        # print("actor loss : {} , character loss: {}".format(actor_loss, character_loss))
        # time.sleep(10)
        # self.recog_optimizer.zero_grad()
        # # character_loss.backward()
        # RMSE_loss.backward()    
        # self.recog_optimizer.step()
        file = open(self.output_dir + '/' + 'character.txt', 'w')
        write_character(file, recog_charater)
        file.write('*******************************\n')
        write_character(file, recog_charater - real_character)
        file.close()

        self.writer.add_scalar(f'{self.tag_name}/loss_character', RMSE_loss.detach().item(), self.step_update)   
        self.writer.add_scalar(f'{self.tag_name}/loss_critic', critic_loss.detach().item(), self.step_update)
        self.writer.add_scalar(f'{self.tag_name}/loss_actor', actor_loss.detach().item(), self.step_update)
        self.writer.add_scalar(f'{self.tag_name}/alpha', self.alpha.detach().item(), self.step_update)

        self._update_model()
        if self.step_update % self.save_model_interval == 0:
            self._save_model()

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
        rllib.utils.soft_update(self.critic_target, self.critic, self.tau)

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

        self.recog = config.get('net_recog', RecognitionWoAttention)(config).to(self.device)

        self.apply(init_weights)

    def forward(self, state):        
        #add character into state
        obs_character = self.recog(state)
        #####
        x = self.fe(state, obs_character)
        mean = self.mean_no(self.mean(x))
        logstd = self.std_no(self.std(x))
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        return mean, logstd *0.5


    def sample(self, state):
        mean, logstd = self(state)

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

class Critic(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        self.recog = config.get('net_actor_recog', RecognitionWoAttention)(config, 0)
        self.fe = config.get('net_critic_fe', FeatureExtractor)(config, 0)
        self.fm1 = config.get('net_critic_fm', FeatureMapper)(config, 0, self.fe.dim_feature+config.dim_action, 1)
        self.fm2 = copy.deepcopy(self.fm1)
        self.apply(init_weights)

    def forward(self, state, action):
        obs_character = self.recog(state)
        #####
        x = self.fe(state, obs_character)
        # x = self.fe(state)
        # breakpoint()
        x = torch.cat([x, action], 1)
        return self.fm1(x), self.fm2(x)
    
    def q1(self, state, action):
        obs_character = self.recog(state)
        #####
        x = self.fe(state, obs_character)
        # x = self.fe(state)
        x = torch.cat([x, action], 1)
        return self.fm1(x)

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