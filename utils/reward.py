import torch
import universe

import numpy as np
import torch



class RewardFunctionNoCharacter(universe.RewardFunc):
    def run_step(self, state, action, agents_master: universe.AgentsMaster, episode_info):
        REWARD_C = -2000
        REWARD_B = -2000
        collision = episode_info.collision
        off_road = episode_info.off_road
        off_route = episode_info.off_route
        wrong_lane = episode_info.wrong_lane

        reward = []
        for i, agent in enumerate(agents_master.vehicles_neural):
            max_velocity = agent.max_velocity

            ### 1. collision
            reward_collision = int(collision[i]) * REWARD_C /100

            ### 2. boundary
            reward_boundary = int(off_road[i] | off_route[i] | wrong_lane[i]) * REWARD_B /100

            ### 3. velocity
            ref_vel = max_velocity * 0.75
            reward_v = (agent.get_state().v - ref_vel) / (max_velocity / 2)
            reward_v = np.clip(reward_v, -1, 1) * 0.1

            reward.append(reward_collision + reward_v + reward_boundary)
        return reward







class RewardFunctionWithCharacter(universe.RewardFunc):
    def run_step(self, state, action, agents_master: universe.AgentsMaster, episode_info):
        reward = RewardFunctionNoCharacter.run_step(self, state, action, agents_master, episode_info)
        reward = np.array(reward, dtype=np.float32)

        masks = np.zeros_like(state[0].vehicle_masks)
        vis = [agent.vi for agent in agents_master.vehicles_neural]
        masks[vis] = 1

        indexs = np.where(masks > 0)
        reward_padded = np.zeros((masks.shape[0],), dtype=reward.dtype)
        reward_padded[indexs] = reward   ### ! warning: corner case: indexs[0].shape != reward.shape

        reward_with_character = []
        for (agent, s, r) in zip(agents_master.vehicles_neural, state, reward):
            character = agent.character
            agent_masks = s.agent_masks *masks
            num_neighbours = agent_masks.sum() -1  ### ! warning: corner case: num_neighbours < surronding vehicles

            r_others = (reward_padded * agent_masks).sum(axis=0) - r
            # import pdb; pdb.set_trace()
            if num_neighbours > 0:
                r_others /= num_neighbours

            reward_with_character.append( np.cos(character*np.pi/2)* r + np.sin(character*np.pi/2)* r_others )

        return reward_with_character






class RewardFunctionGlobalCoordination(universe.RewardFunc):  ### CoPO
    def run_step(self, state, action, agents_master: universe.AgentsMaster, episode_info):
        reward = RewardFunctionNoCharacter.run_step(self, state, action, agents_master, episode_info)
        reward = np.array(reward, dtype=np.float32)

        masks = np.zeros_like(state[0].vehicle_masks)
        vis = [agent.vi for agent in agents_master.vehicles_neural]
        masks[vis] = 1

        indexs = np.where(masks > 0)
        reward_padded = np.zeros((masks.shape[0],), dtype=reward.dtype)
        reward_padded[indexs] = reward   ### ! warning: corner case: indexs[0].shape != reward.shape

        reward_with_character = []
        for (agent, s, r) in zip(agents_master.vehicles_neural, state, reward):
            character = agent.character
            agent_masks = s.agent_masks *masks
            num_neighbours = agent_masks.sum() -1  ### ! warning: corner case: num_neighbours < surronding vehicles

            r_others = (reward_padded * agent_masks).sum(axis=0) - r

            reward_with_character.append( (r + r_others) / (num_neighbours+1) )

        return reward_with_character




class RewardFunctionRecogCharacter(universe.RewardFunc):
    MSEloss = torch.nn.MSELoss()
    def run_step(self, state, action, agents_master: universe.AgentsMaster, episode_info):
        '''single agent'''
        reward = RewardFunctionNoCharacter.run_step(self, state, action, agents_master, episode_info)
        # assert action.shape[0] == 1 and len(state) == 1 \
        #     and len(agents_master.vehicles_neural) == 1 and len(reward) == 1
        true_character = torch.full(action.shape,agents_master.vehicles_rule[0].character)
        RMSEloss = torch.sqrt(self.MSEloss(torch.tensor(action),true_character))
        RMSEloss = np.clip(RMSEloss,0,0.2)
        reward[0] += np.clip(1/np.tan(5*np.pi*RMSEloss), -0.5, 1)        

        return reward

class RewardFunctionRecogCharacterV2(universe.RewardFunc):
    MSEloss = torch.nn.MSELoss()
    def run_step(self, state, action, agents_master: universe.AgentsMaster, episode_info):
        '''single agent'''
        reward = RewardFunctionNoCharacter.run_step(self, state, action, agents_master, episode_info)
        # assert action.shape[0] == 1 and len(state) == 1 \
        #     and len(agents_master.vehicles_neural) == 1 and len(reward) == 1
        valid_len = len(agents_master.vehicles_rule)
        action = action[:,:valid_len]
        true_character = torch.full(action.shape,agents_master.vehicles_rule[0].character)
        RMSEloss = torch.sqrt(self.MSEloss(torch.tensor(action),true_character))
        RMSEloss = np.clip(RMSEloss,0,0.2)
        reward[0] += np.clip(1/np.tan(5*np.pi*RMSEloss), -0.5, 1)        

        return reward