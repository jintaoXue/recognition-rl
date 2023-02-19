from builtins import breakpoint
from turtle import forward
from matplotlib.pyplot import axis
from pyparsing import actions
from pyrsistent import b
import rllib

import numpy as np
import copy

import torch
import torch.nn as nn

import ray
from typing import Dict
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from rllib.template.model import FeatureMapper
################################################################################################################
###### model ###################################################################################################
################################################################################################################

###### recog + load action model################################################################################
################################################################################################################
class RecognitionNet(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        ##########需要加载的参数

        # self.fe = PointNetWithCharactersAgentHistory(config, model_id)
        # self.mean = rllib.template.model.FeatureMapper(config, model_id, self.fe.dim_feature, config.dim_action)
        # self.std = copy.deepcopy(self.mean)
        # self.load_state_dict(torch.load('~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method/INDEPENDENTSAC_V0_Actor_0_866200_.pth'))

        dim_ego_embedding = 128
        dim_character_embedding = 32
        dim_embedding = dim_ego_embedding + dim_character_embedding


        self.dim_embedding = dim_embedding


        self.character_embedding = nn.Linear(1, dim_character_embedding)

        self.ego_embedding = DeepSetModule(self.dim_state.agent, dim_ego_embedding //2)
        self.ego_embedding_v1 = nn.Linear(self.dim_state.agent, dim_ego_embedding //2)

        self.agent_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.agent_embedding_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding //2),
        )

        self.static_embedding = DeepSetModule(self.dim_state.static, dim_embedding)

        self.type_embedding = VectorizedEmbedding(dim_embedding)

        self.global_head_recognition = MultiheadAttentionGlobalHeadRecognition(dim_embedding, out_dim=1, nhead=4, dropout=0.0 if config.evaluate else 0.1)

        self.tanh = nn.Tanh()

        # self.dim_feature = dim_embedding+dim_character_embedding + dim_character_embedding
        # self.actor = torch.load('****.pth')
        # self.load_state_dict(torch.load('~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method/INDEPENDENTSAC_V0_Actor_0_866200_.pth'))
        #todo 

    def forward(self, state: rllib.basic.Data, **kwargs):
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
        # obs_character = state.obs_character[:,:,-1]
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
            self.character_embedding(state.character.unsqueeze(1))
        ], dim=1)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_embedding = torch.cat([
            self.agent_embedding(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs)
        ], dim=2)

        route_embedding = self.static_embedding(route, route_mask)

        lane_embedding = self.static_embedding(lane.flatten(end_dim=1), lane_mask.flatten(end_dim=1))
        lane_embedding = lane_embedding.view(batch_size,num_lanes, self.dim_embedding)

        bound_embedding = self.static_embedding(bound.flatten(end_dim=1), bound_mask.flatten(end_dim=1))
        bound_embedding = bound_embedding.view(batch_size,num_bounds, self.dim_embedding)


        ### global head recognition
        invalid_polys = ~torch.cat([
            ego_mask,
            obs_mask,
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding.unsqueeze(1), obs_embedding, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
        type_embedding = self.type_embedding(state)

        outputs, attns = self.global_head_recognition(all_embs, type_embedding, invalid_polys, num_agents)
        # self.attention = attns.detach().clone().cpu()
        

        outputs = self.tanh(outputs)
        outputs = (1 + self.tanh(outputs))/2
        #(num_agents, batch, 1) -> (batch, num_agents, 1)
        outputs = outputs.transpose(0, 1)

        return outputs

class RecognitionNetNew(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        ##########需要加载的参数

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
        self.ego_embedding_recog = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.ego_embedding_recog_v1 = nn.Linear(self.dim_state.agent, dim_embedding //2)

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
        self.global_head_recognition = MultiheadAttentionGlobalHeadRecognition(dim_embedding + dim_character_embedding,\
             out_dim=dim_embedding + dim_character_embedding, nhead=4, dropout=0.0 if config.evaluate else 0.1)
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
        
        state = sample_state(state)

        batch_size = state.ego.shape[0]
        num_agents = state.obs.shape[1]
        num_lanes = state.lane.shape[1]
        num_bounds = state.bound.shape[1]
        
        ### data generation
        ego = state.ego[:,-1]
        ego_mask = state.ego_mask.to(torch.bool)[:,[-1]]
        obs = state.obs[:,:,-1]
        obs_mask = state.obs_mask[:,:,-1].to(torch.bool)
        # obs_character = state.obs_character[:,:,-1]
        route = state.route
        route_mask = state.route_mask.to(torch.bool)
        lane = state.lane
        lane_mask = state.lane_mask.to(torch.bool)
        bound = state.bound
        bound_mask = state.bound_mask.to(torch.bool)

        ### embedding
        ego_embedding_recog = torch.cat([
            self.ego_embedding_recog(state.ego, state.ego_mask.to(torch.bool)),
            self.ego_embedding_recog_v1(ego),
            self.character_embedding(state.character.unsqueeze(1))
        ], dim=1)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_embedding_recog = torch.cat([
            self.agent_embedding_recog(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, 160 //2),
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
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding_recog.unsqueeze(1), obs_embedding_recog, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
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
        state = cut_state(state)
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

class RecognitionNetNewWoattn(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        ##########需要加载的参数

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
        batch_size = state.ego.shape[0]
        num_agents = state.obs.shape[1]
        num_lanes = state.lane.shape[1]
        num_bounds = state.bound.shape[1]
        
        ### data generation
        ego = state.ego[:,-1]
        ego_mask = state.ego_mask.to(torch.bool)[:,[-1]]
        obs = state.obs[:,:,-1]
        obs_mask = state.obs_mask[:,:,-1].to(torch.bool)
        # obs_character = state.obs_character[:,:,-1]
        route = state.route
        route_mask = state.route_mask.to(torch.bool)
        lane = state.lane
        lane_mask = state.lane_mask.to(torch.bool)
        bound = state.bound
        bound_mask = state.bound_mask.to(torch.bool)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_embedding_recog = torch.cat([
            self.agent_embedding_recog(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, 160 //2),
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
        # state_ = cut_state(state)

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
class RecognitionWoAttention(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        ##########需要加载的参数

        # self.fe = PointNetWithCharactersAgentHistory(config, model_id)
        # self.mean = rllib.template.model.FeatureMapper(config, model_id, self.fe.dim_feature, config.dim_action)
        # self.std = copy.deepcopy(self.mean)
        # self.load_state_dict(torch.load('~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method/INDEPENDENTSAC_V0_Actor_0_866200_.pth'))

        dim_ego_embedding = 128
        dim_character_embedding = 32
        dim_embedding = dim_ego_embedding + dim_character_embedding


        self.dim_embedding = dim_embedding


        self.agent_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.agent_embedding_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding //2),
        )
        # self.global_head_recognition = MultiheadAttentionGlobalHeadRecognition(dim_embedding, out_dim=1, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.out_proj = NonDynamicallyQuantizableLinear(dim_embedding, out_features = 1, bias=True)
        self.tanh = nn.Tanh()

        # self.dim_feature = dim_embedding+dim_character_embedding + dim_character_embedding
        # self.actor = torch.load('****.pth')
        # self.load_state_dict(torch.load('~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method/INDEPENDENTSAC_V0_Actor_0_866200_.pth'))
        #todo 

    def forward(self, state: rllib.basic.Data, **kwargs):
        # breakpoint()
        batch_size = state.ego.shape[0]
        num_agents = state.obs.shape[1]
        num_lanes = state.lane.shape[1]
        num_bounds = state.bound.shape[1]
        
        ### data generation

        obs = state.obs[:,:,-1]
        obs_mask = state.obs_mask[:,:,-1].to(torch.bool)
        obs_character = state.obs_character[:,:,-1]


        ### embedding

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_embedding = torch.cat([
            self.agent_embedding(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs)
        ], dim=2)


        outputs = self.out_proj(obs_embedding)
        outputs = self.tanh(outputs)
        outputs = (1 + self.tanh(outputs))/2
        #(num_agents, batch, 1) -> (batch, num_agents, 1)
        # outputs = outputs.transpose(0, 1)

        return outputs

'''for the Env to load and evaluate'''
class PointNetWithCharactersAgentHistoryRecog(rllib.template.Model):
    #如果state 的 horizon 大于10
    def __init__(self, config, model_id):
        super().__init__(config, model_id)

        dim_embedding = 128
        dim_character_embedding = 32
        self.dim_embedding = dim_embedding
        self.dim_character_embedding = dim_character_embedding

        self.character_embedding = nn.Linear(1, dim_character_embedding)

        self.ego_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.ego_embedding_v1 = nn.Linear(self.dim_state.agent, dim_embedding //2)

        self.agent_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.agent_embedding_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding //2),
        )

        self.static_embedding = DeepSetModule(self.dim_state.static, dim_embedding +dim_character_embedding)

        self.type_embedding = VectorizedEmbedding(dim_embedding +dim_character_embedding)
        self.global_head = MultiheadAttentionGlobalHead(dim_embedding +dim_character_embedding, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.dim_feature = dim_embedding+dim_character_embedding + dim_character_embedding


    def forward(self, state: rllib.basic.Data, obs_character : torch.Tensor ,**kwargs):
        _state = cut_state(state)
        batch_size = _state.ego.shape[0]
        num_agents = _state.obs.shape[1]
        num_lanes = _state.lane.shape[1]
        num_bounds = _state.bound.shape[1]

        ### data generation
        ego = _state.ego[:,-1]
        ego_mask = _state.ego_mask.to(torch.bool)[:,[-1]]
        obs = _state.obs[:,:,-1]
        obs_mask = _state.obs_mask[:,:,-1].to(torch.bool)
        # obs_character = state.obs_character[:,:,-1]
        route = _state.route
        route_mask = _state.route_mask.to(torch.bool)
        lane = _state.lane
        lane_mask = _state.lane_mask.to(torch.bool)
        bound = _state.bound
        bound_mask = _state.bound_mask.to(torch.bool)

        ### embedding
        ego_embedding = torch.cat([
            self.ego_embedding(_state.ego, _state.ego_mask.to(torch.bool)),
            self.ego_embedding_v1(ego),
            self.character_embedding(_state.character.unsqueeze(1)),
        ], dim=1)


        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_character = torch.where(obs_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=obs.device), obs_character)

        obs_embedding = torch.cat([
            self.agent_embedding(_state.obs.flatten(end_dim=1), _state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs),
            self.character_embedding(obs_character),
        ], dim=2)

        route_embedding = self.static_embedding(route, route_mask)

        lane_embedding = self.static_embedding(lane.flatten(end_dim=1), lane_mask.flatten(end_dim=1))
        lane_embedding = lane_embedding.view(batch_size,num_lanes, self.dim_embedding + self.dim_character_embedding)

        bound_embedding = self.static_embedding(bound.flatten(end_dim=1), bound_mask.flatten(end_dim=1))
        bound_embedding = bound_embedding.view(batch_size,num_bounds, self.dim_embedding + self.dim_character_embedding)


        ### global head
        invalid_polys = ~torch.cat([
            ego_mask,
            obs_mask,
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding.unsqueeze(1), obs_embedding, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
        type_embedding = self.type_embedding(_state)
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)
        self.attention = attns.detach().clone().cpu()

        outputs = torch.cat([outputs, self.character_embedding(_state.character.unsqueeze(1))], dim=1)
        return outputs

###### recog as action #########################################################################################
################################################################################################################
'''output a single svo(all the obstacles have same svos)'''
class RecogNetSVO(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        # self.dim_action = 20
        dim_ego_embedding = 128
        dim_character_embedding = 32
        dim_embedding = dim_ego_embedding + dim_character_embedding

        self.dim_embedding = dim_embedding


        self.character_embedding = nn.Linear(1, dim_character_embedding)

        self.ego_embedding = DeepSetModule(self.dim_state.agent, dim_ego_embedding //2)
        self.ego_embedding_v1 = nn.Linear(self.dim_state.agent, dim_ego_embedding //2)

        self.agent_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.agent_embedding_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding //2),
        )

        self.static_embedding = DeepSetModule(self.dim_state.static, dim_embedding)

        self.type_embedding = VectorizedEmbedding(dim_embedding)

        self.global_head_recognition = MultiheadAttentionGlobalHead(dim_embedding, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.dim_feature = dim_embedding+dim_character_embedding

    def forward(self, state: rllib.basic.Data, **kwargs):
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
        # obs_character = state.obs_character[:,:,-1]
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
            self.character_embedding(state.character.unsqueeze(1))
        ], dim=1)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_embedding = torch.cat([
            self.agent_embedding(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs)
        ], dim=2)

        route_embedding = self.static_embedding(route, route_mask)

        lane_embedding = self.static_embedding(lane.flatten(end_dim=1), lane_mask.flatten(end_dim=1))
        lane_embedding = lane_embedding.view(batch_size,num_lanes, self.dim_embedding)

        bound_embedding = self.static_embedding(bound.flatten(end_dim=1), bound_mask.flatten(end_dim=1))
        bound_embedding = bound_embedding.view(batch_size,num_bounds, self.dim_embedding)

        ### global head recognition
        invalid_polys = ~torch.cat([
            ego_mask,
            obs_mask,
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding.unsqueeze(1), obs_embedding, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
        type_embedding = self.type_embedding(state)

        outputs, _ = self.global_head_recognition(all_embs, type_embedding, invalid_polys)
        # self.attention = attns.detach().clone().cpu()
        outputs = torch.cat([outputs, self.character_embedding(state.character.unsqueeze(1))], dim=1)
        return outputs

class RecogNetSvoWoattn(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        ##########需要加载的参数
        dim_embedding = 128
        dim_character_embedding = 32
        self.dim_embedding_agent = dim_embedding + dim_character_embedding

        self.character_embedding = nn.Linear(1, dim_character_embedding)
        self.agent_embedding = DeepSetModule(self.dim_state.agent, self.dim_embedding_agent//2)
        self.agent_embedding_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, self.dim_embedding_agent), nn.ReLU(inplace=True),
            nn.Linear(self.dim_embedding_agent, self.dim_embedding_agent), nn.ReLU(inplace=True),
            nn.Linear(self.dim_embedding_agent, self.dim_embedding_agent //2),
        )
        # self.global_head_recognition = MultiheadAttentionGlobalHeadRecognition(dim_embedding, out_dim=1, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        # self.out_proj = NonDynamicallyQuantizableLinear(self.dim_embedding_agent, out_features = 1, bias=True)

        self.dim_feature = self.dim_embedding_agent + dim_character_embedding

    def forward(self, state: rllib.basic.Data, **kwargs):
        # breakpoint()
        batch_size = state.ego.shape[0]
        num_agents = state.obs.shape[1]
        ### data generation
        obs = state.obs[:,:,-1]
        # obs_mask = state.obs_mask[:,:,-1].to(torch.bool)
        # obs_character = state.obs_character[:,:,-1]
        ### embedding
        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        outputs = torch.cat([
            self.agent_embedding(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding_agent //2),
            self.agent_embedding_v1(obs)
        ], dim=2)

        if outputs.shape[1] != 0: outputs = outputs.mean(dim=1)
        else :
            outputs = torch.full([batch_size,self.dim_embedding_agent], -1, dtype=torch.float32).to(self.device)
        character_embedding = self.character_embedding(state.character.unsqueeze(1))
        outputs = torch.cat([outputs, character_embedding], dim=1)
        return outputs
'''output Multi svo(all the obstacles have same/different svos)'''
class RecogNetMultiSVO(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        # self.dim_action = 20
        dim_ego_embedding = 128
        dim_character_embedding = 32
        dim_embedding = dim_ego_embedding + dim_character_embedding

        self.dim_embedding = dim_embedding


        self.character_embedding = nn.Linear(1, dim_character_embedding)

        self.ego_embedding = DeepSetModule(self.dim_state.agent, dim_ego_embedding //2)
        self.ego_embedding_v1 = nn.Linear(self.dim_state.agent, dim_ego_embedding //2)

        self.agent_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.agent_embedding_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding //2),
        )

        self.static_embedding = DeepSetModule(self.dim_state.static, dim_embedding)

        self.type_embedding = VectorizedEmbedding(dim_embedding)

        self.global_head_recognition = MultiheadAttentionGlobalHeadRecognition(dim_embedding, \
            dim_embedding,nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.dim_feature = dim_embedding+dim_character_embedding

    def forward(self, state: rllib.basic.Data, **kwargs):
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
        # obs_character = state.obs_character[:,:,-1]
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
            self.character_embedding(state.character.unsqueeze(1))
        ], dim=1)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_embedding = torch.cat([
            self.agent_embedding(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs)
        ], dim=2)

        route_embedding = self.static_embedding(route, route_mask)

        lane_embedding = self.static_embedding(lane.flatten(end_dim=1), lane_mask.flatten(end_dim=1))
        lane_embedding = lane_embedding.view(batch_size,num_lanes, self.dim_embedding)

        bound_embedding = self.static_embedding(bound.flatten(end_dim=1), bound_mask.flatten(end_dim=1))
        bound_embedding = bound_embedding.view(batch_size,num_bounds, self.dim_embedding)

        ### global head recognition
        invalid_polys = ~torch.cat([
            ego_mask,
            obs_mask,
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding.unsqueeze(1), obs_embedding, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
        type_embedding = self.type_embedding(state)
        outputs, attns = self.global_head_recognition(all_embs, type_embedding, invalid_polys, num_agents)

        # outputs = torch.where(outputs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)
        outputs = outputs.transpose(0, 1)
        # self.attention = attns.detach().clone().cpu()
        #[batch_size, num_agents, dim]
        character_embed = self.character_embedding(state.character.unsqueeze(1)).unsqueeze(1).repeat(1,num_agents,1)         
        # if(len(outputs.shape) == 2 or len(character_embed.shape) == 2):breakpoint()
        outputs = torch.cat([outputs, character_embed], dim=2)
        return outputs
class RecogNetCritic(rllib.template.Model):
    pass
class RecogNetMultiSVOWoattn(rllib.template.Model):
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        ##########需要加载的参数

        # self.fe = PointNetWithCharactersAgentHistory(config, model_id)
        # self.mean = rllib.template.model.FeatureMapper(config, model_id, self.fe.dim_feature, config.dim_action)
        # self.std = copy.deepcopy(self.mean)
        # self.load_state_dict(torch.load('~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method/INDEPENDENTSAC_V0_Actor_0_866200_.pth'))

        dim_ego_embedding = 128
        dim_character_embedding = 32
        dim_embedding = dim_ego_embedding + dim_character_embedding


        self.dim_embedding = dim_embedding


        self.agent_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.agent_embedding_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding //2),
        )
        # self.global_head_recognition = MultiheadAttentionGlobalHeadRecognition(dim_embedding, out_dim=1, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.out_proj = NonDynamicallyQuantizableLinear(dim_embedding, out_features = dim_embedding, bias=True)

        self.dim_feature = dim_embedding

    def forward(self, state: rllib.basic.Data, **kwargs):
        # breakpoint()
        batch_size = state.ego.shape[0]
        num_agents = state.obs.shape[1]
        ### data generation
        obs = state.obs[:,:,-1]
        # obs_mask = state.obs_mask[:,:,-1].to(torch.bool)
        # obs_character = state.obs_character[:,:,-1]
        ### embedding
        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_embedding = torch.cat([
            self.agent_embedding(state.obs.flatten(end_dim=1), state.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs)
        ], dim=2)


        outputs = self.out_proj(obs_embedding)
        #(num_agents, batch, 1) -> (batch, num_agents, 1)
        # outputs = outputs.transpose(0, 1)

        return outputs

'''for Env to load the net'''
class PointNetwithActionSVO(rllib.template.Model):
    #作为action为svo的网络
    def __init__(self, config, model_id):
        super().__init__(config, model_id)

        dim_embedding = 128
        dim_character_embedding = 32
        self.dim_embedding = dim_embedding
        self.dim_character_embedding = dim_character_embedding

        self.character_embedding = nn.Linear(1, dim_character_embedding)

        self.ego_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.ego_embedding_v1 = nn.Linear(self.dim_state.agent, dim_embedding //2)

        self.agent_embedding = DeepSetModule(self.dim_state.agent, dim_embedding //2)
        self.agent_embedding_v1 = nn.Sequential(
            nn.Linear(self.dim_state.agent, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding), nn.ReLU(inplace=True),
            nn.Linear(dim_embedding, dim_embedding //2),
        )

        self.static_embedding = DeepSetModule(self.dim_state.static, dim_embedding +dim_character_embedding)

        self.type_embedding = VectorizedEmbedding(dim_embedding +dim_character_embedding)
        self.global_head = MultiheadAttentionGlobalHead(dim_embedding +dim_character_embedding, nhead=4, dropout=0.0 if config.evaluate else 0.1)
        self.dim_feature = dim_embedding+dim_character_embedding + dim_character_embedding

    def forward(self, state: rllib.basic.Data):
        state_ = cut_state(state)
        batch_size = state_.ego.shape[0]
        num_agents = state_.obs.shape[1]
        num_lanes = state_.lane.shape[1]
        num_bounds = state_.bound.shape[1]

        ### data generation
        ego = state_.ego[:,-1]
        ego_mask = state_.ego_mask.to(torch.bool)[:,[-1]]
        obs = state_.obs[:,:,-1]
        obs_mask = state_.obs_mask[:,:,-1].to(torch.bool)
        obs_character = state_.obs_character[:,:,-1]
        route = state_.route
        route_mask = state_.route_mask.to(torch.bool)
        lane = state_.lane
        lane_mask = state_.lane_mask.to(torch.bool)
        bound = state_.bound
        bound_mask = state_.bound_mask.to(torch.bool)

        ### embedding
        ego_embedding = torch.cat([
            self.ego_embedding(state_.ego, state_.ego_mask.to(torch.bool)),
            self.ego_embedding_v1(ego),
            self.character_embedding(state_.character.unsqueeze(1)),
        ], dim=1)


        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)

        obs_character = torch.where(obs_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=obs.device), obs_character)

        obs_embedding = torch.cat([
            self.agent_embedding(state_.obs.flatten(end_dim=1), state_.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs),
            self.character_embedding(obs_character),
        ], dim=2)

        route_embedding = self.static_embedding(route, route_mask)

        lane_embedding = self.static_embedding(lane.flatten(end_dim=1), lane_mask.flatten(end_dim=1))
        lane_embedding = lane_embedding.view(batch_size,num_lanes, self.dim_embedding + self.dim_character_embedding)

        bound_embedding = self.static_embedding(bound.flatten(end_dim=1), bound_mask.flatten(end_dim=1))
        bound_embedding = bound_embedding.view(batch_size,num_bounds, self.dim_embedding + self.dim_character_embedding)


        ### global head
        invalid_polys = ~torch.cat([
            ego_mask,
            obs_mask,
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding.unsqueeze(1), obs_embedding, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
        type_embedding = self.type_embedding(state_)
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)
        # self.attention = attns.detach().clone().cpu()

        outputs = torch.cat([outputs, self.character_embedding(state_.character.unsqueeze(1))], dim=1)
        return outputs

    def forward_with_svo(self, state: rllib.basic.Data, obs_character : np.ndarray ,**kwargs):
        #从 agent master类收到的感知信息
        state_ = cut_state(state)
        # state_ = state_.to(self.device)
        # obs_character = torch.from_numpy(obs_character).to(self.device)
        # print('agent master obs_svos', obs_character.shape)
        batch_size = state_.ego.shape[0]
        num_agents = state_.obs.shape[1]
        num_lanes = state_.lane.shape[1]
        num_bounds = state_.bound.shape[1]

        ### data generation
        ego = state_.ego[:,-1]

        ego_mask = state_.ego_mask.to(torch.bool)[:,[-1]]
        obs = state_.obs[:,:,-1]
        obs_mask = state_.obs_mask[:,:,-1].to(torch.bool)
        # obs_character = state.obs_character[:,:,-1]
        route = state_.route
        route_mask = state_.route_mask.to(torch.bool)
        lane = state_.lane
        lane_mask = state_.lane_mask.to(torch.bool)
        bound = state_.bound
        bound_mask = state_.bound_mask.to(torch.bool)

        ### embedding
        ego_embedding = torch.cat([
            self.ego_embedding(state_.ego, state_.ego_mask.to(torch.bool)),
            self.ego_embedding_v1(ego),
            self.character_embedding(state_.character.unsqueeze(1)),
        ], dim=1)

        obs = torch.where(obs == np.inf, torch.tensor(0, dtype=torch.float32, device=obs.device), obs)
        obs_character = torch.where(obs_character == np.inf, torch.tensor(-1, dtype=torch.float32, device=obs.device), obs_character)
        # if obs.shape[0] != obs_character.shape[0]:breakpoint()
        # if obs.shape[1] != obs_character.shape[1]:breakpoint()
        obs_embedding = torch.cat([
            self.agent_embedding(state_.obs.flatten(end_dim=1), state_.obs_mask.to(torch.bool).flatten(end_dim=1)).view(batch_size,num_agents, self.dim_embedding //2),
            self.agent_embedding_v1(obs),
            self.character_embedding(obs_character),
        ], dim=2)

        route_embedding = self.static_embedding(route, route_mask)

        lane_embedding = self.static_embedding(lane.flatten(end_dim=1), lane_mask.flatten(end_dim=1))
        lane_embedding = lane_embedding.view(batch_size,num_lanes, self.dim_embedding + self.dim_character_embedding)

        bound_embedding = self.static_embedding(bound.flatten(end_dim=1), bound_mask.flatten(end_dim=1))
        bound_embedding = bound_embedding.view(batch_size,num_bounds, self.dim_embedding + self.dim_character_embedding)


        ### global head
        invalid_polys = ~torch.cat([
            ego_mask,
            obs_mask,
            route_mask.any(dim=1, keepdim=True),
            lane_mask.any(dim=2),
            bound_mask.any(dim=2),
        ], dim=1)
        all_embs = torch.cat([ego_embedding.unsqueeze(1), obs_embedding, route_embedding.unsqueeze(1), lane_embedding, bound_embedding], dim=1)
        type_embedding = self.type_embedding(state_)
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)
        # self.attention = attns.detach().clone().cpu()

        outputs = torch.cat([outputs, self.character_embedding(state_.character.unsqueeze(1))], dim=1)
        return outputs


################################################################################################################
###### buffer ##################################################################################################
################################################################################################################


def pad_state_with_characters(state: rllib.basic.Data):
    def pad(element, pad_value):
        sizes = torch.tensor([list(e.shape) for e in element])
        max_sizes = torch.Size(sizes.max(dim=0).values)
        return [pad_data(e, max_sizes, pad_value) for e in element]

    obs = pad(state.obs, pad_value=np.inf)
    obs_mask = pad(state.obs_mask, pad_value=0)
    obs_character = pad(state.obs_character, pad_value=np.inf)

    lane = pad(state.lane, pad_value=np.inf)
    lane_mask = pad(state.lane_mask, pad_value=0)

    bound = pad(state.bound, pad_value=np.inf)
    bound_mask = pad(state.bound_mask, pad_value=0)

    state.update(obs=obs, obs_mask=obs_mask, obs_character=obs_character, lane=lane, lane_mask=lane_mask, bound=bound, bound_mask=bound_mask)
    return


from utils import buffer


class ReplayBufferSingleAgentMultiWorker(buffer.ReplayBufferMultiWorker):
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



################################################################################################################
###### basic ###################################################################################################
################################################################################################################
from rllib.template.model import FeatureExtractor, FeatureMapper
from rllib.utils import init_weights
from torch.distributions import Normal, MultivariateNormal

class Actor(rllib.template.Model):
    logstd_min = -5
    logstd_max = 1

    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)
        # self.recog = config.get('net_actor_recog', RecognitionNet)(config, 0)
        self.mean_no = nn.Tanh()
        self.std_no = nn.Tanh()
        #todo
        self.fe = config.get('net_actor_fe', FeatureExtractor)(config, 0)
        self.mean = config.get('net_actor_fm', FeatureMapper)(config, 0, self.fe.dim_feature, config.dim_action)
        self.std = copy.deepcopy(self.mean)
        self.apply(init_weights)

    def forward(self, state):
        #add character into state
        # obs_character = self.recog(state)
        # print(obs_character)
        #####
        x = self.fe(state)
        mean = self.mean_no(self.mean(x))
        logstd = self.std_no(self.std(x))
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        return mean, logstd *0.5
    
    def forward_with_svo(self, state, obs_svos):
        x = self.fe.forward_with_svo(state, obs_svos)
        mean = self.mean_no(self.mean(x))
        # logstd = self.std_no(self.std(x))
        # logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        return mean

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


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
class DeepSetModule(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.phi = nn.Sequential(
            nn.Linear(dim_in, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )
        self.rho = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, dim_out),
        )

    def forward(self, x, mask):
        """
            x: torch.Size([batch_size, dim_elements, dim_feature])
            mask: torch.Size([batch_size, dim_elements])
        """

        batch_size, total_length = x.shape[0], x.shape[1]
        sorted_mask, sorted_mask_index = torch.sort(mask.to(torch.int64), dim=1, descending=True)

        lengths = sorted_mask.to(torch.int64).sum(dim=1)
        valid_index = torch.where(lengths != 0)
        valid_lengths = lengths[valid_index].cpu()
        valid = len(valid_lengths) > 0

        res = torch.full((batch_size, self.dim_out), 0.0, dtype=x.dtype, device=x.device)
        if valid:
            sorted_x = torch.gather(x, 1, sorted_mask_index.unsqueeze(2).repeat(1,1,self.dim_in))
            x_pack = pack_padded_sequence(sorted_x[valid_index], valid_lengths, batch_first=True, enforce_sorted=False)
            valid_x = self.phi(x_pack.data)
            valid_x = PackedSequence(valid_x, x_pack.batch_sizes, x_pack.sorted_indices, x_pack.unsorted_indices)
            valid_x, _ = pad_packed_sequence(valid_x, batch_first=True, padding_value=0, total_length=total_length)
            valid_x = torch.sum(valid_x, dim=1)
            valid_x = self.rho(valid_x)

            res[valid_index] = valid_x
        return res

class VectorizedEmbedding(nn.Module):
    def __init__(self, dim_embedding: int):
        """A module which associates learnable embeddings to types

        :param dim_embedding: features of the embedding
        :type dim_embedding: int
        """
        super().__init__()

        self.polyline_types = {
            "AGENT_OF_INTEREST": 0,
            "AGENT_NO": 1,
            "AGENT_CAR": 2,
            "ROUTE": 3,
            "LANE_CENTER": 4,
            "BOUND": 5,
        }
        self.dim_embedding = dim_embedding
        self.embedding = nn.Embedding(len(self.polyline_types), dim_embedding)

        # Torch script did like dicts as Tensor selectors, so we are going more primitive.
        # self.PERCEPTION_LABEL_CAR: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
        # self.PERCEPTION_LABEL_PEDESTRIAN: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_PEDESTRIAN"]
        # self.PERCEPTION_LABEL_CYCLIST: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CYCLIST"]

    def forward(self, state: rllib.basic.Data):
        """
        Model forward: embed the given elements based on their type.

        Assumptions:
        - agent of interest is the first one in the batch
        - other agents follow
        - then we have polylines (lanes)
        """

        with torch.no_grad():
            batch_size = state.ego.shape[0]

            other_agents_len = state.obs.shape[1]
            route_len = 1
            lanes_len = state.lane.shape[1]
            bounds_len = state.bound.shape[1]
            total_len = 1 + other_agents_len + route_len + lanes_len + bounds_len

            other_agents_start_idx = 1
            route_start_index = other_agents_start_idx + other_agents_len
            lanes_start_idx = route_start_index + route_len
            bounds_start_idx = lanes_start_idx + lanes_len

            indices = torch.full(
                (batch_size, total_len),
                fill_value=self.polyline_types["AGENT_NO"],
                dtype=torch.long,
                device=state.ego.device,
            )

            indices[:, 0].fill_(self.polyline_types["AGENT_OF_INTEREST"])
            indices[:, other_agents_start_idx:route_start_index].fill_(self.polyline_types["AGENT_CAR"])
            indices[:, route_start_index:lanes_start_idx].fill_(self.polyline_types["ROUTE"])

            indices[:, lanes_start_idx:bounds_start_idx].fill_(self.polyline_types["LANE_CENTER"])
            indices[:, bounds_start_idx:].fill_(self.polyline_types["BOUND"])


        return self.embedding.forward(indices)

class MultiheadAttentionGlobalHeadRecognition(nn.Module):
    def __init__(self, d_model, out_dim, nhead=8, dropout=0.1):
        super().__init__()
        from .attn import MultiheadAttention
        self.encoder = MultiheadAttention(d_model, out_dim, nhead, dropout=dropout)

    def forward(self, inputs: torch.Tensor, type_embedding: torch.Tensor, mask: torch.Tensor, num_agents):
        inputs = inputs.transpose(0, 1)
        type_embedding = type_embedding.transpose(0, 1)
        outputs, attns = self.encoder(inputs[1:num_agents+1], inputs + type_embedding, inputs, mask)
        # return torch.cat([outputs, inputs[[0]]], dim=2), attns
        # return outputs + inputs[[0]], attns
        # return outputs, attns
        return outputs, attns

class MultiheadAttentionGlobalHead(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.encoder = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, inputs: torch.Tensor, type_embedding: torch.Tensor, mask: torch.Tensor):
        inputs = inputs.transpose(0, 1)
        type_embedding = type_embedding.transpose(0, 1)
        outputs, attns = self.encoder(inputs[[0]], inputs + type_embedding, inputs, mask)
        # return torch.cat([outputs, inputs[[0]]], dim=2), attns
        # return outputs + inputs[[0]], attns
        return outputs.squeeze(0), attns.squeeze(1)

def pad_data(data: torch.Tensor, pad_size: torch.Size, pad_value=np.inf):
    """
    Args:
        data, pad_size: torch.Size([batch_size, dim_elements, dim_points, dim_features])
    """
    res = torch.full(pad_size, pad_value, dtype=data.dtype, device=data.device)

    if len(pad_size) == 2:
        batch_size, dim_elements = data.shape
        res[:batch_size, :dim_elements] = data
    elif len(pad_size) == 3:
        batch_size, dim_elements, dim_points = data.shape
        res[:batch_size, :dim_elements, :dim_points] = data
    elif len(pad_size) == 4:
        batch_size, dim_elements, dim_points, dim_features = data.shape
        res[:batch_size, :dim_elements, :dim_points, :dim_features] = data
    else:
        raise NotImplementedError
    return res

def cut_state(state: rllib.basic.Data) :
    state_ = copy.deepcopy(state)
    horizon = 10
    raw_horizon = 30
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

def sample_state(state: rllib.basic.Data) :
    state_ = copy.deepcopy(state)
    #hrz30 -> 10
    horizon = 10
    raw_horizon = 30
    interval = int(raw_horizon / horizon)
    state_.ego = torch.cat((state_.ego[:,interval-1:-1:interval,:], state_.ego[:,-1:,:]), 1)  
    state_.ego_mask = torch.cat((state_.ego_mask[:,interval-1:-1:interval], state_.ego_mask[:,-1:]), 1) 

    state_.obs = torch.cat((state_.obs[:,:,interval-1:-1:interval,:], state_.obs[:,:,-1:,:]), 2)  
    state_.obs_mask = torch.cat((state_.obs_mask[:,:,interval-1:-1:interval], state_.obs_mask[:,:,-1:]), 2)  
    # state_.obs_character = state_.obs_character[:,:,-horizon:,-1]
    return state_


