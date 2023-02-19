from random import sample
import rllib

import numpy as np
import copy

import torch
import torch.nn as nn

import ray
from typing import Dict
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from rllib.template.model import FeatureMapper
from core.recognition_net import DeepSetModule, MultiheadAttentionGlobalHeadRecognition, RecognitionNetNew, \
                            MultiheadAttentionGlobalHead, VectorizedEmbedding

class RecognitionNetNewWoMap(RecognitionNetNew):
    def forward(self, state: rllib.basic.Data, **kwargs):
        # breakpoint()
        
        state_sampled = sample_state(state, self.raw_horizon, self.sampled_horizon)

        batch_size = state.ego.shape[0]
        num_agents = state.obs.shape[1]
        num_lanes = state.lane.shape[1]
        num_bounds = state.bound.shape[1]
        
        ### data generation
        ego = state_sampled.ego[:,-1]
        ego_mask = state_sampled.ego_mask.to(torch.bool)[:,[-1]]
        obs = state_sampled.obs[:,:,-1]
        obs_mask = state_sampled.obs_mask[:,:,-1].to(torch.bool)
        # obs_character = state.obs_character[:,:,-1]
        route = state.route
        route_mask = state.route_mask.to(torch.bool)
        lane = state.lane
        lane_mask = state.lane_mask.to(torch.bool)
        bound = state.bound
        bound_mask = state.bound_mask.to(torch.bool)

        ### embedding
        ego_embedding_recog = torch.cat([
            self.ego_embedding_recog(state_sampled.ego, state_sampled.ego_mask.to(torch.bool)),
            self.ego_embedding_recog_v1(ego),
            self.character_embedding(state_sampled.character.unsqueeze(1))
        ], dim=1)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_embedding_recog = torch.cat([
            self.agent_embedding_recog(state_sampled.obs.flatten(end_dim=1), state_sampled.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, 160 //2),
            self.agent_embedding_recog_v1(obs)
        ], dim=2)

        route_embedding = self.static_embedding(route, route_mask)

        lane_embedding = self.static_embedding(lane.flatten(end_dim=1), lane_mask.flatten(end_dim=1))
        lane_embedding = lane_embedding.view(batch_size,num_lanes, self.dim_embedding + self.dim_character_embedding)

        bound_embedding = self.static_embedding(bound.flatten(end_dim=1), bound_mask.flatten(end_dim=1))
        bound_embedding = bound_embedding.view(batch_size,num_bounds, self.dim_embedding + self.dim_character_embedding)

        ### global head recognition
        invalid_polys_recog = ~torch.cat([
            ego_mask,
            obs_mask,
        ], dim=1)
        all_embs = torch.cat([ego_embedding_recog.unsqueeze(1), obs_embedding_recog], dim=1)
        type_embedding = self.type_embedding(state)
        obs_svos, attns = self.global_head_recognition(all_embs, type_embedding, invalid_polys_recog, num_agents)
        # breakpoint()
        # obs_svos = (1 + self.tanh(obs_svos))/2
        obs_svos = self.tanh(self.recog_feature_mapper(obs_svos))
        #(num_agents, batch, 1) -> (batch, num_agents, 1)
        obs_svos = obs_svos.transpose(0, 1)
        self.obs_svos = obs_svos
        # state_ = cut_state(state)
        #debug 
        state = cut_state(state, self.raw_horizon, 10)
        ego = state.ego[:,-1]
        ego_mask = state.ego_mask.to(torch.bool)[:,[-1]]
        obs = state.obs[:,:,-1]
        obs_mask = state.obs_mask[:,:,-1].to(torch.bool)
        ego_embedding = torch.cat([
            self.ego_embedding(state.ego, state.ego_mask.to(torch.bool)),
            self.ego_embedding_v1(ego),
            self.character_embedding(state.character.unsqueeze(1)),
        ], dim=1)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)
        obs_character = obs_svos
        obs_embedding = torch.cat([
            self.agent_embedding(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs),
            self.character_embedding(obs_character),
        ], dim=2)

        ### global head
        invalid_polys = ~torch.cat([
            ego_mask,
            obs_mask,
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding.unsqueeze(1), obs_embedding, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)
        self.attention = attns.detach().clone().cpu()

        outputs = torch.cat([outputs, self.character_embedding(state.character.unsqueeze(1))], dim=1)
        return outputs



class RecognitionNetNewWoattn(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        ##########需要加载的参数
        self.raw_horizon = config.raw_horizon
        self.sampled_horizon = config.horizon 
        # self.fe = PointNetWithCharactersAgentHistory(config, model_id)
        # self.mean = rllib.template.model.FeatureMapper(config, model_id, self.fe.dim_feature, config.dim_action)
        # self.std = copy.deepcopy(self.mean)
        # self.load_state_dict(torch.load('~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method/INDEPENDENTSAC_V0_Actor_0_866200_.pth'))
        dim_embedding = 128
        dim_character_embedding = 32
        self.dim_embedding = dim_embedding
        self.dim_character_embedding = dim_character_embedding

        self.character_embedding = nn.Linear(1, dim_character_embedding)
        #recognition
        # self.ego_embedding_recog = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        # self.ego_embedding_recog_v1 = nn.Linear(self.dim_state.agent, dim_embedding //2)

        self.agent_embedding_recog = DeepSetModule(self.dim_state.agent, 160 //2)
        self.agent_embedding_recog_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, 160), nn.ReLU(inplace=True),
            nn.Linear(160, 160), nn.ReLU(inplace=True),
            nn.Linear(160, 160 //2),
        )
        #action
        self.ego_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.ego_embedding_v1 = nn.Linear(self.dim_state.agent, dim_embedding //2)

        self.agent_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.agent_embedding_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding //2),
        )

        self.static_embedding = DeepSetModule(self.dim_state.static, dim_embedding +dim_character_embedding)
        self.type_embedding = VectorizedEmbedding(dim_embedding + dim_character_embedding)

        # self.global_head_recognition = MultiheadAttentionGlobalHeadRecognition(dim_embedding, out_dim=1, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.out_proj = NonDynamicallyQuantizableLinear(dim_embedding + dim_character_embedding, out_features = dim_embedding+dim_character_embedding, bias=True)

        # self.global_head_recognition = MultiheadAttentionGlobalHead(dim_embedding + dim_character_embedding, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.recog_feature_mapper = FeatureMapper(config, model_id, dim_embedding + dim_character_embedding, 1)
        self.tanh = nn.Tanh()
        # self.dim_feature = dim_embedding + dim_character_embedding + dim_character_embedding
        # self.actor = torch.load('****.pth')
        # self.load_state_dict(torch.load('~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method/INDEPENDENTSAC_V0_Actor_0_866200_.pth'))
        #todo 
        self.global_head = MultiheadAttentionGlobalHead(dim_embedding +dim_character_embedding, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.dim_feature = dim_embedding+dim_character_embedding + dim_character_embedding
        self.obs_svos = torch.empty(1,1)
    def get_recog_obs_svos(self):
        return self.obs_svos
    def forward(self, state: rllib.basic.Data, **kwargs):
        # breakpoint()

        _state = sample(state, self.raw_horizon, self.sampled_horizon)
        batch_size = state.ego.shape[0]
        num_agents = state.obs.shape[1]
        num_lanes = state.lane.shape[1]
        num_bounds = state.bound.shape[1]
        
        ### data generation
        ego = _state.ego[:,-1]
        ego_mask = _state.ego_mask.to(torch.bool)[:,[-1]]
        obs = _state.obs[:,:,-1]
        obs_mask = _state.obs_mask[:,:,-1].to(torch.bool)

        # obs_character = state.obs_character[:,:,-1]
        route = state.route
        route_mask = state.route_mask.to(torch.bool)
        lane = state.lane
        lane_mask = state.lane_mask.to(torch.bool)
        bound = state.bound
        bound_mask = state.bound_mask.to(torch.bool)

        obs = torch.where(_state == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_embedding_recog = torch.cat([
            self.agent_embedding_recog(_state.obs.flatten(end_dim=1), _state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, 160 //2),
            self.agent_embedding_recog_v1(obs)
        ], dim=2)

        route_embedding = self.static_embedding(route, route_mask)

        lane_embedding = self.static_embedding(lane.flatten(end_dim=1), lane_mask.flatten(end_dim=1))
        lane_embedding = lane_embedding.view(batch_size,num_lanes, self.dim_embedding + self.dim_character_embedding)

        bound_embedding = self.static_embedding(bound.flatten(end_dim=1), bound_mask.flatten(end_dim=1))
        bound_embedding = bound_embedding.view(batch_size,num_bounds, self.dim_embedding + self.dim_character_embedding)
        type_embedding = self.type_embedding(state)
        ### recognition

        outputs = self.out_proj(obs_embedding_recog)

    
        # obs_svos = (1 + self.tanh(obs_svos))/2
        obs_svos = self.tanh(self.recog_feature_mapper(outputs))
        #(num_agents, batch, 1) -> (batch, num_agents, 1)
        self.obs_svos = obs_svos
        state = cut_state(state, self.raw_horizon, 10)
        ego = state.ego[:,-1]
        ego_mask = state.ego_mask.to(torch.bool)[:,[-1]]
        obs = state.obs[:,:,-1]
        obs_mask = state.obs_mask[:,:,-1].to(torch.bool)
        #####step two : action####
        ### embedding
        ego_embedding = torch.cat([
            self.ego_embedding(state.ego, state.ego_mask.to(torch.bool)),
            self.ego_embedding_v1(ego),
            self.character_embedding(state.character.unsqueeze(1)),
        ], dim=1)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)
        obs_character = obs_svos
        obs_embedding = torch.cat([
            self.agent_embedding(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs),
            self.character_embedding(obs_character),
        ], dim=2)

        ### global head
        invalid_polys = ~torch.cat([
            ego_mask,
            obs_mask,
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding.unsqueeze(1), obs_embedding, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)
        self.attention = attns.detach().clone().cpu()

        outputs = torch.cat([outputs, self.character_embedding(state.character.unsqueeze(1))], dim=1)
        return outputs

    def forward_with_true_svo(self, state: rllib.basic.Data, **kwargs):
        # breakpoint()
        batch_size = state.ego.shape[0]
        num_agents = state.obs.shape[1]
        num_lanes = state.lane.shape[1]
        num_bounds = state.bound.shape[1]
        
        ### data generation
        ego = state.ego[:,-1]
        ego_mask = state.ego_mask.to(torch.bool)[:,[-1]]
        obs = state.obs[:,:,-1]
        obs_mask = state.obs_mask[:,:,-1].to(torch.bool)
        obs_character = state.obs_character[:,:,-1]
        route = state.route
        route_mask = state.route_mask.to(torch.bool)
        lane = state.lane
        lane_mask = state.lane_mask.to(torch.bool)
        bound = state.bound
        bound_mask = state.bound_mask.to(torch.bool)
        ### embedding
        ego_embedding = torch.cat([
            self.ego_embedding(state.ego, state.ego_mask.to(torch.bool)),
            self.ego_embedding_v1(ego),
            self.character_embedding(state.character.unsqueeze(1)),
        ], dim=1)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)
        obs_character = torch.where(obs_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=obs.device), obs_character)
        obs_embedding = torch.cat([
            self.agent_embedding(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs),
            self.character_embedding(obs_character),
        ], dim=2)
        route_embedding = self.static_embedding(route, route_mask)

        lane_embedding = self.static_embedding(lane.flatten(end_dim=1), lane_mask.flatten(end_dim=1))
        lane_embedding = lane_embedding.view(batch_size,num_lanes, self.dim_embedding + self.dim_character_embedding)

        bound_embedding = self.static_embedding(bound.flatten(end_dim=1), bound_mask.flatten(end_dim=1))
        bound_embedding = bound_embedding.view(batch_size,num_bounds, self.dim_embedding + self.dim_character_embedding)
        type_embedding = self.type_embedding(state)
        ### global head
        invalid_polys = ~torch.cat([
            ego_mask,
            obs_mask,
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding.unsqueeze(1), obs_embedding, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)
        self.attention = attns.detach().clone().cpu()

        outputs = torch.cat([outputs, self.character_embedding(state.character.unsqueeze(1))], dim=1)
        return outputs




def cut_state(state: rllib.basic.Data, raw_horizon, horizon) :
    state_ = copy.deepcopy(state)
    state_.ego = state_.ego[:,-horizon:,:]
    #change horizon stamp from raw_horizon to custom horizon
    state_.ego[...,-1] = state_.ego[...,-1]*raw_horizon/horizon
    state_.ego_mask = state_.ego_mask[:,-horizon:]

    state_.obs = state_.obs[:,:,-horizon:,:]
    #change horizon stamp from raw_horizon to custom horizon
    state_.obs[...,-1] = state_.obs[...,-1]*raw_horizon/horizon
    state_.obs_mask = state_.obs_mask[:,:,-horizon:]
    # state_.obs_character = state_.obs_character[:,:,-horizon:,-1]
    return state_

def sample_state(state: rllib.basic.Data, raw_horizon, horizon) :
    state_ = copy.deepcopy(state)
    interval = int(raw_horizon / horizon)
    state_.ego = torch.cat((state_.ego[:,interval-1:-1:interval,:], state_.ego[:,-1:,:]), 1)  
    state_.ego_mask = torch.cat((state_.ego_mask[:,interval-1:-1:interval], state_.ego_mask[:,-1:]), 1) 

    state_.obs = torch.cat((state_.obs[:,:,interval-1:-1:interval,:], state_.obs[:,:,-1:,:]), 2)  
    state_.obs_mask = torch.cat((state_.obs_mask[:,:,interval-1:-1:interval], state_.obs_mask[:,:,-1:]), 2)  
    # state_.obs_character = state_.obs_character[:,:,-horizon:,-1]
    return state_