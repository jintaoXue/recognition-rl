

from builtins import breakpoint
import copy
from turtle import update
from importlib_metadata import re

import numpy as np
import time

import os, glob
from os.path import join
import torch
import torch.nn as nn
from torch.optim import Adam

from torch.distributions import Normal, MultivariateNormal

from rllib.utils import init_weights

import rllib
from rllib.basic import prefix, Data as Experience
# from rllib.buffer.tools import stack_data
from rllib.template import MethodSingleAgent, Model
from rllib.template.model import FeatureExtractor, FeatureMapper

from core.recognition_net import RecognitionNet
from .model_vectornet import RolloutBufferSingleAgentMultiWorker, pad_state_with_characters
import pickle
from typing import List

class IndependentSACsupervise(MethodSingleAgent):
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
    start_timesteps = 0
    # start_timesteps = 1000  ## ! warning
    before_training_steps = 0

    save_model_interval = 10000


    def __init__(self, config: rllib.basic.YamlConfig, writer):
        super().__init__(config, writer)

        self.actor = config.get('net_actor', Actor)(config).to(self.device)

        #todo
        self.actor.method_name = 'INDEPENDENTSAC_V0'
        # self.actor.model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
        # self.actor.model_num = 865800

        self.actor.model_dir = '~/github/zdk/recognition-rl/models/origin_no_history_bottleneck/'
        self.actor.model_num = 445600
        self.models_to_load = [self.actor]
        # [model.load_model() for model in self.models_to_load]
        [load_model(model) for model in self.models_to_load]
        self.actor.method_name = 'IndependentSACsupervise'

        self.training_data_path = config.training_data_path

        self.data_size = 2019
        #todo
        self.actor_target = copy.deepcopy(self.actor)
        self.models_to_save = [self.actor]

        self.recog_optimizer= Adam(self.actor.recog.parameters(), lr=self.lr_actor)
        
        self.recog_loss = nn.MSELoss()#l1 loss
        # for name,param in self.actor.state_dict(keep_vars=True).items():
        #     print(name,param.requires_grad)     
        ### automatic entropy tuning

        # to adpat the child class method : IndependentSACsuperviseRoll which have initialized RolloutBuffer*
        if config.get('buffer') is RolloutBufferSingleAgentMultiWorker: 
            return
        self.buffer: rllib.buffer.ReplayBuffer = config.get('buffer', rllib.buffer.ReplayBuffer)(config, self.buffer_size, self.batch_size, self.device)

    def update_parameters(self):
        '''load data batch'''
        t1 = time.time()
        indices = np.random.randint(0, self.data_size, size=self.batch_size)
        batch_data: List[Experience] = []
        for indice in indices :
            # file_path = os.path.join(self.training_data_path+f'{indice}.txt')
            # '~/dataset/carla/scenario_offline/bottleneck/1.txt'
            file_path = os.path.join(self.training_data_path, f'{indice}.txt')
            with open(file_path, 'rb') as f: 
                batch_data.append(pickle.load(f))
        experience = self._batch_stack(batch_data)
        state = experience.state.to(self.device)
        t2 = time.time()
        t = t2 - t1
        # if len(self.buffer) < self.start_timesteps + self.before_training_steps:
        #     return
        self.update_parameters_start()

        '''character MSE'''
        t1 = time.time()
        recog_character = self.actor.recog(state)  
        t2 = time.time()
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

        self.writer.add_scalar(f'{self.tag_name}/loss_character',  RMSE_loss.detach().item(), self.step_update)   
        self.writer.add_scalar(f'{self.tag_name}/recog_time', t2-t1, self.step_update)
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

    def _batch_stack(self, batch):
        result = rllib.buffer.stack_data(batch)

        state, next_state = result.state, result.next_state

        self.pad_state(state)
        self.pad_state(next_state)
        state.pop('agent_masks')
        state.pop('vehicle_masks')
        next_state.pop('agent_masks')
        next_state.pop('vehicle_masks')
        result.update(state=state)
        result.update(next_state=next_state)

        result = result.cat(dim=0)
        result.vi.unsqueeze_(1)
        result.reward.unsqueeze_(1)
        result.done.unsqueeze_(1)
        return result

    def pad_state(self, state: rllib.basic.Data):
        pad_state_with_characters(state)



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

        self.recog = config.get('net_recog', RecognitionNet)(config).to(self.device)

        self.apply(init_weights)

        #evaluate
        self.recog_loss = nn.MSELoss()

    def forward(self, state):        
        #add character into state
        obs_character = self.recog(state)

        # #evaluate
        # real_character = state.obs_character[:,:,-1]
        # recog_character = obs_character[~torch.isinf(real_character)]
        # real_character = real_character[~torch.isinf(real_character)]
        
        # real_character = torch.where(real_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), real_character)
        # recog_character = torch.where(recog_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), recog_character)
        # character_loss = self.recog_loss(recog_character, real_character)
        # print("****************loss{}".format( torch.sqrt(character_loss)))
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

    def close(self):
        return


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