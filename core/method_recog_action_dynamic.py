

from cmath import nan
import copy
from multiprocessing.connection import wait
from timeit import timeit
from traceback import print_tb
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

from core.recognition_net import RecognitionNet, DeepSetModule
import time

class RecogV2(MethodSingleAgent):
    dim_reward = 2
    
    gamma = 0.9

    target_entropy = None
    alpha_init = 1.0

    lr_critic = 5e-4
    lr_actor = 1e-4
    lr_tune = 1e-4

    tau = 0.005

    buffer_size = 750000
    batch_size = 128

    start_timesteps = 30000
    # start_timesteps = 0  ## ! warning
    before_training_steps = 0

    save_model_interval = 1000
    print_svo_mse_interval = 100
    def __init__(self, config: rllib.basic.YamlConfig, writer):
        super().__init__(config, writer)

        self.critic = config.get('net_critic', Critic)(config).to(self.device)
        self.actor = config.get('net_actor', Actor)(config).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)
        self.models_to_save = [self.actor, self.critic]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)

        self.critic_loss = nn.MSELoss()
        self.character_loss = nn.MSELoss()
        #same other svo
        self.dim_action = 19
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
        # print('critic_loss: {}'.format(critic_loss))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''actor'''
        action, logprob, _ = self.actor.sample(state)
        # actor_loss = (-self.critic.q1(state, action) + self.alpha * logprob).mean() * self.actor_loss_scale
        # breakpoint()
        actor_loss = ((-self.critic.q1(state, action) + self.alpha * logprob).mean())
        # actor_loss = torch.nn.init.uniform(actor_loss, a=0, b=1)
        # print('-self.critic.q1(state, action) :{}, self.alpha * logprob:{}\n'.format(-self.critic.q1(state, action) , self.alpha * logprob))
        # print('actor_loss : {}'.format(actor_loss) ,actor_loss)
        self.actor_optimizer.zero_grad()
        # for name, parms in self.actor.named_parameters():	
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #     print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:',parms.grad)
        #     print("===")
        actor_loss.backward()
        # for name, parms in self.actor.named_parameters():	
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #     print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:',parms.grad)
        #     print("===")
        # breakpoint()
        # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1)
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
        if self.step_update % self.print_svo_mse_interval == 0:
            with torch.no_grad():
                recog_charater, _= self.actor(state) 
            real_character = state.obs_character[:,:,-1]
            valid_len = real_character.shape[1]
            recog_charater = recog_charater[:,:valid_len]
            recog_charater = torch.where(real_character == np.inf, torch.tensor(np.inf, dtype=torch.float32, device=state.obs.device),recog_charater)
            real_character = real_character[~torch.isinf(real_character)]
            recog_charater = recog_charater[~torch.isinf(recog_charater)]
            # breakpoint()
            # real_character = torch.where(real_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), real_character)
            # recog_charater = torch.where(recog_charater == np.inf, torch.tensor(-1, dtype=torch.float32, device=state.obs.device), recog_charater)
            with torch.no_grad():
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
            action = torch.Tensor(len(state), self.dim_action).uniform_(0,1)
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
            valid_len = state.obs_character.shape[1]
            action = torch.Tensor(1, self.dim_action,1).uniform_(0,1)
            action[0,valid_len:] = -1
        else:
            action, _, _ = self.actor.sample(state.to(self.device))
            action = action.cpu()
        # print('select_action', action.shape)
        return action
    
    def _update_model(self):
        # print('[update_parameters] soft update')
        rllib.utils.soft_update(self.critic_target, self.critic, self.tau)



class Actor(rllib.template.Model):
    logstd_min = -5
    logstd_max = 1

    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        # self.recog = config.get('net_actor_recog', RecognitionNet)(config, 0)
        self.mean_no = nn.Tanh()
        self.std_no = nn.Tanh()
        #todo
        config.set('dim_action', 1)
        self.max_other_vehicles = 19
        self.fe = config.get('net_actor_fe', FeatureExtractor)(config, 0)
        self.mean = config.get('net_actor_fm', FeatureMapper)(config, 0 ,self.fe.dim_feature, config.dim_action)
        self.std = copy.deepcopy(self.mean)
        self.apply(init_weights)

    def forward(self, state):
        #[batch_size, num_agents, dim]
        x = self.fe(state)
        num_svo = x.shape[1]
        mean = self.mean(x)
        mean = self.mean_no(mean)
        logstd = self.std_no(self.std(x))
        #to do
        mean = torch.cat([mean, torch.full((len(x), \
            self.max_other_vehicles- num_svo, self.dim_action), -1.0).to(self.device)], dim = 1)
        logstd = torch.cat([logstd, torch.full((len(x), \
            self.max_other_vehicles- num_svo,self.dim_action), -1.0).to(self.device)], dim = 1)
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        #torch.isnan(x).any()
        # if torch.isnan(mean).any() :
        #     print('_______________________')
        #     breakpoint()
        # print('forward', mean.shape)
        # breakpoint()
        return mean, logstd *0.5


    def sample(self, state):

        mean, logstd= self(state)
        num_svo = state.obs_character.shape[1]
        cov = torch.diag_embed(torch.exp(logstd))
        dist = MultivariateNormal(mean, cov)
        u = dist.rsample()
        # if mean.shape[0] == 1:
        #     print('    policy entropy: ', dist.entropy().detach().cpu())
        #     print('    policy mean:    ', mean.detach().cpu())
        #     print('    policy std:     ', torch.exp(logstd).detach().cpu())
        ### Enforcing Action Bound
        action = torch.tanh(u)
        logprob = dist.log_prob(u)[:,:num_svo].sum(-1).unsqueeze(1) \
                - torch.log(1 - action[:,:num_svo].pow(2) + 1e-6).sum(dim=1)
        # print('sample', action.shape)
        
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
        config.set('dim_action', 1)
        self.dim_action = 1
        # self.recog = config.get('net_actor_recog', RecognitionNet)(config, 0)
        self.fe = config.get('net_critic_fe', FeatureExtractor)(config, 0)
        self.fm1 = config.get('net_critic_fm', FeatureMapper)(config, 0, self.fe.dim_feature+self.dim_action, 1)
        self.fm2 = copy.deepcopy(self.fm1)
        self.apply(init_weights)

    def forward(self, state, action):
        # obs_character = self.recog(state)
        #####
        # x = self.fe(state, obs_character)
        x = self.fe(state)
        num_svo = x.shape[1]
        action = action[:,0:num_svo]
        x = torch.cat([x,action], dim=2)
        return torch.mean(self.fm1(x),dim=1), torch.mean(self.fm2(x),dim=1)
    
    def q1(self, state, action):
        # obs_character = self.recog(state)
        #####
        # x = self.fe(state, obs_character)
        x = self.fe(state)
        num_svo = x.shape[1]
        action = action[:,0:num_svo]
        x = torch.cat([x,action], dim=2)
        return torch.mean(x, dim=1)
        # x = self.fe(state)
        # num_svo = x.shape[1]
        # list_fm1 = []

        # for i in range(0, num_svo):
        #     x_ = torch.cat([x[:,i],action[:,i:i+1]], dim=1)
        #     list_fm1.append(self.fm1(x_))

        # return torch.mean(torch.cat(list_fm1, dim=1),dim=1).unsqueeze(1)
    # def forward(self, state, action):
    #     x = self.fe(state)
    #     x = torch.cat([x, action], 1)
    #     return self.fm1(x), self.fm2(x)
    
    # def q1(self, state, action):
    #     x = self.fe(state)
    #     x = torch.cat([x, action], 1)
    #     return self.fm1(x)
def write_character(file, character) :
    character = character.detach().cpu().numpy()
    # character = [item.detach().cpu().numpy() for item in character]
    # character = character.squeeze(2)
    #将numpy类型转化为list类型
    character=character.tolist()
    #将list转化为string类型
    str1=str(character)
    file.write(str1 + '\n\n')

    

