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

class RecognitionNetNewWomap(RecognitionNetNew):

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

        obs_svos, attns = self.global_head_recognition(all_embs, type_embedding[:,:num_agents+1], invalid_polys_recog, num_agents)
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



class RecognitionNetNewWoattn(RecognitionNetNew):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        del self.global_head_recognition 
        # self.global_head_recognition = MultiheadAttentionGlobalHead(dim_embedding + dim_character_embedding, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.recog_feature_mapper = FeatureMapper(config, model_id, self.dim_embedding, 1)

    def forward(self, state: rllib.basic.Data, **kwargs):
        # breakpoint()
        state_sampled = sample_state(state, self.raw_horizon, self.sampled_horizon)
        batch_size = state.ego.shape[0]
        num_agents = state.obs.shape[1]
        num_lanes = state.lane.shape[1]
        num_bounds = state.bound.shape[1]
        
        ### data generation

        obs = state_sampled.obs[:,:,-1]
        obs_mask = state_sampled.obs_mask[:,:,-1].to(torch.bool)
        obs_character = state_sampled.obs_character[:,:,-1]
        ### embedding

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)
        obs_embedding_recog = torch.cat([
            self.agent_embedding(state_sampled.obs.flatten(end_dim=1), state_sampled.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs)
        ], dim=2)

        obs_svos = self.recog_feature_mapper(obs_embedding_recog)
        obs_svos = self.tanh(obs_svos)

        self.obs_svos = obs_svos

        state = cut_state(state, self.raw_horizon, 10)
        route = state.route
        route_mask = state.route_mask.to(torch.bool)
        lane = state.lane
        lane_mask = state.lane_mask.to(torch.bool)
        bound = state.bound
        bound_mask = state.bound_mask.to(torch.bool)
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

