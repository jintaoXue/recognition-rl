import rllib

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence



################################################################################################################
###### method ##################################################################################################
################################################################################################################



class TD3(rllib.td3.TD3):
    gamma = 0.9
    
    lr_critic = 5e-4
    lr_critic = 2e-4
    lr_actor = 1e-4

    tau = 0.005

    buffer_size = 1000000
    batch_size = 128

    policy_freq = 4
    explore_noise = 0.1
    policy_noise = 0.2
    noise_clip = 0.4

    start_timesteps = 30000
    start_timesteps = 10000
    # start_timesteps = 1000  ## ! warning
    # start_timesteps = 5000  ## ! warning

    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()

        if self.step_select < self.start_timesteps:
            action = torch.Tensor(1,self.dim_action).uniform_(-1,1)
        else:
            noise = torch.normal(0, self.explore_noise, size=(1,self.dim_action))
            action = self.actor(state.to(self.device))
            action = (action.cpu() + noise).clamp(-1,1)
        return action


    def _save_model(self):
        return





################################################################################################################
###### model ###################################################################################################
################################################################################################################



class DeepSetOld(rllib.template.Model):
    def __init__(self, config, model_id):
        super().__init__(config, model_id)
        dim_in = self.dim_state['obs']
        dim_out = 64
        self.dim_in, self.dim_out = dim_in, dim_out

        self.phi = nn.Sequential(
            nn.Linear(dim_in, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512),
        )
        self.rho = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, dim_out),
        )
        self.dim_feature = dim_out + self.dim_state['ego'] + self.dim_state['ref']

    def forward(self, state, **kwargs):
        ### x:    torch.Size([batch_size, length+1, dim])
        ### mask: torch.Size([batch_size, length+1])
        x, mask = state.dynamic, state.mask

        batch_size, total_length = x.shape[0], x.shape[1]-1
        result = torch.zeros((batch_size, self.dim_out), dtype=x.dtype, device=x.device)

        sorted_mask, sorted_mask_index = torch.sort(mask, dim=1, descending=True)

        lengths = sorted_mask.sum(dim=1) + 1
        valid_index = torch.where(lengths != 0)
        valid_lengths = lengths[valid_index].cpu()
        valid = len(valid_lengths) > 0

        if valid:
            x = torch.gather(x, 1, sorted_mask_index.unsqueeze(-1).repeat(1,1,self.dim_in))[:,:-1]
            x_pack = pack_padded_sequence(x[valid_index], valid_lengths, batch_first=True, enforce_sorted=False)
            x = self.phi(x_pack.data)
            x = PackedSequence(x, x_pack.batch_sizes, x_pack.sorted_indices, x_pack.unsorted_indices)
            x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0, total_length=total_length)
            x = torch.sum(x, dim=1)
            # x = torch.max(x, dim=1).values
            x = self.rho(x)
            result[valid_index] = x

        result = torch.cat([state.fixed, state.ref, result], dim=1)
        return result



class DeepSet(rllib.template.Model):
    """
        Only for single agent.
    """
    def __init__(self, config, model_id):
        super().__init__(config, model_id)

        dim_in = self.dim_state['obs']
        dim_out = 64
        self.deepset = DeepSetModule(dim_in, dim_out)

        self.dim_feature = dim_out + self.dim_state['ego']


    def forward(self, state: rllib.basic.Data, **kwargs):
        # ego = state.ego[:,-1][:,2:6]
        # x = state.obs[:,:,-1][:,:,2:6]
        # mask = state.obs_mask[:,:,-1].to(torch.bool)

        ego = state.fixed
        x = state.dynamic[:,1:]
        mask = state.mask[:,1:].to(torch.bool)

        res = self.deepset(x, mask)
        res = torch.cat([ego, res], dim=1)
        return res



class DeepSetWithCharacter(rllib.template.Model):
    def __init__(self, config, model_id):
        super().__init__(config, model_id)
        dim_in = self.dim_state['obs']
        dim_out = 64
        self.dim_in, self.dim_out = dim_in, dim_out

        # self.phi = nn.Sequential(
        #     nn.Linear(dim_in, 128), nn.ReLU(inplace=True),
        #     nn.Linear(128, 256), nn.ReLU(inplace=True),
        #     nn.Linear(256, 512),
        # )
        # self.rho = nn.Sequential(
        #     nn.Linear(512, 256), nn.ReLU(inplace=True),
        #     nn.Linear(256, 128), nn.ReLU(inplace=True),
        #     nn.Linear(128, dim_out),
        # )
        self.deepset = DeepSetModule(dim_in, dim_out)

        dim_character_encoding = 128
        self.character_encoding = nn.Linear(1, dim_character_encoding)
        self.dim_feature = dim_out + self.dim_state['ego'] + self.dim_state['ref'] + dim_character_encoding

    def forward(self, state):
        ### x:    torch.Size([batch_size, length+1, dim])
        ### mask: torch.Size([batch_size, length+1])
        x, mask = state.dynamic, state.mask
        mask = torch.where(mask == -1, 0, mask).to(torch.bool)


        # import pdb; pdb.set_trace()
        # print('here')

        result = self.deepset(x, mask)

        # batch_size, total_length = x.shape[0], x.shape[1]-1
        # result = torch.zeros((batch_size, self.dim_out), dtype=x.dtype, device=x.device)

        # sorted_mask, sorted_mask_index = torch.sort(mask, dim=1, descending=True)

        # lengths = sorted_mask.sum(dim=1) + 1
        # valid_index = torch.where(lengths != 0)
        # valid_lengths = lengths[valid_index].cpu()
        # valid = len(valid_lengths) > 0

        # if valid:
        #     x = torch.gather(x, 1, sorted_mask_index.unsqueeze(-1).repeat(1,1,self.dim_in))[:,:-1]
        #     x_pack = pack_padded_sequence(x[valid_index], valid_lengths, batch_first=True, enforce_sorted=False)
        #     x = self.phi(x_pack.data)
        #     x = PackedSequence(x, x_pack.batch_sizes, x_pack.sorted_indices, x_pack.unsorted_indices)
        #     x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0, total_length=total_length)
        #     x = torch.sum(x, dim=1)
        #     x = self.rho(x)
        #     result[valid_index] = x
        

        character = state.fixed[:,[7]]
        character_encoding = self.character_encoding(character)

        result = torch.cat([character_encoding, state.fixed, state.ref, result], dim=1)
        return result





################################################################################################################
###### buffer ##################################################################################################
################################################################################################################



class ReplayBuffer(rllib.buffer.ReplayBuffer):
    def _batch_stack(self, batch):

        result = rllib.buffer.stack_data(batch)
        result.pop('timestamp')
        result = result.cat(dim=0)
        result.reward.unsqueeze_(1)
        result.done.unsqueeze_(1)
        return result



from utils import buffer
class ReplayBufferMultiProcess(buffer.ReplayBufferMultiProcess):
    def _batch_stack(self, batch):

        result = rllib.buffer.stack_data(batch)
        result.pop('timestamp')
        result = result.cat(dim=0)
        result.reward.unsqueeze_(1)
        result.done.unsqueeze_(1)
        return result



class RolloutBuffer(rllib.buffer.RolloutBuffer):
    def _batch_stack(self, batch):

        result = rllib.buffer.stack_data(batch)
        result.pop('timestamp')
        result = result.cat(dim=0)
        result.reward.unsqueeze_(1)
        result.done.unsqueeze_(1)
        return result





class ReplayBufferIndependent(rllib.buffer.ReplayBuffer):
    def push(self, experiences):
        for experience in experiences:
            self.memory[self.size % self.capacity] = experience
            self.size += 1
        return

    def _batch_stack(self, batch):
        result = rllib.buffer.stack_data(batch)
        result.pop('timestamp')

        result_final = result.cat(dim=0)

        result_final.update(actions=torch.stack(result.actions, dim=0))
        result_final.update(rewards=torch.stack(result.rewards, dim=0))
        result_final.update(dones=torch.stack(result.dones, dim=0))
        result_final.update(masks=torch.stack(result.masks, dim=0))

        result_final.vi.unsqueeze_(1)
        result_final.character.unsqueeze_(1)

        return result_final










################################################################################################################
###### basic ###################################################################################################
################################################################################################################


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
class DeepSetModule(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        # self.phi = nn.Sequential(
        #     nn.Linear(dim_in, 256), nn.ReLU(inplace=True),
        #     nn.Linear(256, 256), nn.ReLU(inplace=True),
        #     nn.Linear(256, 512),
        # )
        # self.rho = nn.Sequential(
        #     nn.Linear(512, 256), nn.ReLU(inplace=True),
        #     nn.Linear(256, 256), nn.ReLU(inplace=True),
        #     nn.Linear(256, dim_out),
        # )
        self.phi = nn.Sequential(
            nn.Linear(dim_in, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512),
        )
        self.rho = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, dim_out),
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


