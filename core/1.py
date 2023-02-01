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