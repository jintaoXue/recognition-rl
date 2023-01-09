

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

from core.recognition_net import RecognitionNet
import time

class RecogV1(MethodSingleAgent):
    dim_reward = 2
    
    gamma = 0.9

    target_entropy = None
    alpha_init = 1.0

    lr_critic = 5e-4
    lr_actor = 1e-4
    lr_tune = 0.5e-4

    tau = 0.005

    buffer_size = 750000
    batch_size = 32

    start_timesteps = 50000
    start_timesteps = 100  ## ! warning
    before_training_steps = 0

    save_model_interval = 1000
    print_svo_mse_interval = 10
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
        self.dim_action = 1
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
        print(self.critic.q1(state, action), 'logprob', logprob)
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
                recog_charater, _ = self.actor(state) 
    
            real_character = state.obs_character[:,0,-1]
            
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
            action = torch.Tensor(1,self.dim_action).uniform_(0,1)
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
        # self.recog = config.get('net_actor_recog', RecognitionNet)(config, 0)
        self.mean_no = nn.Tanh()
        self.std_no = nn.Tanh()
        #todo
        config.set('dim_action', 1)
        self.dim_action = 1
        self.fe = config.get('net_actor_fe', FeatureExtractor)(config, 0)
        self.mean = config.get('net_actor_fm', FeatureMapper)(config, 0, self.fe.dim_feature, config.dim_action)
        self.std = copy.deepcopy(self.mean)
        self.apply(init_weights)

    def forward(self, state):
        x = self.fe(state)
        mean = self.mean_no(self.mean(x))
        logstd = self.std_no(self.std(x))
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        #torch.isnan(x).any()
        # if torch.isnan(mean).any() :
        #     print('_______________________')
        #     breakpoint()
        return mean, logstd *0.5


    def sample(self, state):

        mean, logstd= self(state)
        cov = torch.diag_embed(torch.exp(logstd))
        dist = MultivariateNormal(mean, cov)
        u = dist.rsample()
        # if mean.shape[0] == 1:
        #     print('    policy entropy: ', dist.entropy().detach().cpu())
        #     print('    policy mean:    ', mean.detach().cpu())
        #     print('    policy std:     ', torch.exp(logstd).detach().cpu())
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
        # x = self.fe(state)
        # breakpoint()
        x = torch.cat([x, action], 1)
        return self.fm1(x), self.fm2(x)
    
    def q1(self, state, action):
        # obs_character = self.recog(state)
        #####
        # x = self.fe(state, obs_character)
        x = self.fe(state)
        x = torch.cat([x, action], 1)
        return self.fm1(x)

def write_character(file, character) :
    character = character.detach().cpu().numpy()
    # character = [item.detach().cpu().numpy() for item in character]
    # character = character.squeeze(2)
    #将numpy类型转化为list类型
    character=character.tolist()
    #将list转化为string类型
    str1=str(character)
    file.write(str1 + '\n\n')

    
# def init_weights(m):
#     if isinstance(m, (nn.Conv2d, nn.Linear)):
#         nn.init.uniform_(m.weight)
#         try: nn.init.constant_(m.bias, 0.01)
#         except: pass
#     if isinstance(m, nn.LSTM):
#         for name, param in m.named_parameters():
#             if name.startswith('weight'): nn.init.orthogonal_(param)
#     return




class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] == "Softmax", "Final actor layer must be softmax"
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                           key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)

        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        assert not self.hyperparameters["add_extra_noise"], "There is no add extra noise option for the discrete version of SAC at moment"
        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilitie