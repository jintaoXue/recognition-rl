

from builtins import breakpoint
import copy
from turtle import update
from charset_normalizer import utils
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
from rllib.basic import prefix
from rllib.template import MethodSingleAgent, Model
from rllib.template.model import FeatureExtractor, FeatureMapper
from zmq import device

from core.recognition_net import RecognitionNet
from .model_vectornet import RolloutBufferSingleAgentMultiWorker, pad_state_with_characters

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

    buffer_size = 800000
    batch_size = 128

    start_timesteps = 800000
    # start_timesteps = 1  ## ! warning
    before_training_steps = 0

    save_model_interval = 10000


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
        if config.get('buffer') is RolloutBufferSingleAgentMultiWorker: 
            return
        self.buffer: rllib.buffer.ReplayBuffer = ReplayBufferMultiAgentMultiWorker(config, self.buffer_size, self.batch_size, self.device)

    def update_parameters(self):
        if len(self.buffer) < self.start_timesteps + self.before_training_steps:
            return
        self.update_parameters_start()
        self.writer.add_scalar(f'{self.tag_name}/buffer_size', len(self.buffer), self.step_update)

        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state
        # action = experience.action
        # next_state = experience.next_state
        # reward = experience.reward
        # done = experience.done

        '''character MSE'''
        t1 = time.time()
        self.actor.fe(state)  
        recog_character = self.actor.fe.get_recog_obs_svos()
        t2 = time.time()
        real_character = state.obs_character[:,:,-1]
        recog_character = recog_character[~torch.isinf(real_character)]
        real_character = real_character[~torch.isinf(real_character)]
        
        # breakpoint()
        # real_character = torch.where(real_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), real_character)
        # recog_character = torch.where(recog_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), recog_character)
        character_loss = self.recog_loss(recog_character, real_character)
        RMSE_loss = torch.sqrt(character_loss)
        self.actor_optimizer.zero_grad()
        # character_loss.backward()
        RMSE_loss.backward()    
        self.actor_optimizer.step()
        # for name,p in self.actor.fe.named_parameters():
        #     print(name, p)  
        # time.sleep(10)
        file = open(self.output_dir + '/' + 'character.txt', 'w')
        write_character(file, recog_character)
        write_character(file, real_character)
        write_character(file, recog_character - real_character)
        file.write('*******************************\n')
        file.close()

        self.writer.add_scalar(f'{self.tag_name}/loss_character',  RMSE_loss.detach().item(), self.step_update)   
        self.writer.add_scalar(f'{self.tag_name}/recog_time', t2-t1, self.step_update)
        # self.writer.add_scalar(f'{self.tag_name}/alpha', self.alpha.detach().item(), self.step_update)

        # self._update_model()
        if self.step_update % self.save_model_interval == 0:
            self._save_model()

        return


    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()
        # print('select: ', self.step_select)
        states = rllib.buffer.stack_data(state)
        self.buffer.pad_state(states)
        states = states.cat(dim=0)
        action = self.actor.sample(states.to(self.device))
        action = action.cpu()
        return action

    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        action = self.actor.sample(state.to(self.device))
        action = action.cpu()
        return action

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
        x = self.fe.forward_with_true_svo(state)
        mean = self.mean_no(self.mean(x))
        logstd = self.std_no(self.std(x))
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        return mean, logstd *0.5

    def sample(self, state):
        mean, logstd = self(state)
        return mean




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

class ReplayBuffer(rllib.buffer.ReplayBuffer):
    def push_new(self, experience, **kwargs):
        breakpoint()
        rate = 0.2
        index = self.size % self.capacity
        if self.size >= self.capacity:
            index = (index % int((1-rate)*self.capacity)) + rate*self.capacity
            index = int(index)

        self.memory[index] = experience
        self.size += 1
    
    def push(self, experience, **kwargs):
        if self.size >= self.capacity : 
            return
        self.memory[self.size % self.capacity] = experience
        self.size += 1

class ReplayBufferMultiWorker(object):
    def __init__(self, config, capacity, batch_size, device):
        num_workers = config.num_workers
        self.num_workers = num_workers
        self.batch_size, self.device = batch_size, device
        self.buffers = {i: ReplayBuffer(config, capacity //num_workers, batch_size, device) for i in range(num_workers)}
        return

    def __len__(self):
        lengths = [len(b) for b in self.buffers.values()]
        return sum(lengths)

    def push(self, experience, **kwargs):
        i = kwargs.get('index')
        self.buffers[i].push(experience)
        return

    def sample(self):
        batch_sizes = rllib.basic.split_integer(self.batch_size, self.num_workers)
        batch = []
        for i, buffer in self.buffers.items():
            batch.append(buffer.get_batch(batch_sizes[i]))
        batch = np.concatenate(batch)
        return self._batch_stack(batch).to(self.device)

    def _batch_stack(self, batch):
        raise NotImplementedError

class ReplayBufferMultiAgentMultiWorker(ReplayBufferMultiWorker):
    
    def push(self, experience, **kwargs):
        i = kwargs.get('index')
        for e in experience:
            self.buffers[i].push(e)
        return

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

