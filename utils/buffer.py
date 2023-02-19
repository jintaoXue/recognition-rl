import rllib

import numpy as np
import torch
import copy
from rllib.basic import Data as Experience

class ReplayBufferMultiWorker(object):
    def __init__(self, config, capacity, batch_size, device):
        num_workers = config.num_workers
        self.num_workers = num_workers
        self.capacity = capacity
        self.batch_size, self.device = batch_size, device
        self.buffers = {i: ReplayBuffer(config, capacity //num_workers, batch_size, device) for i in range(num_workers)}
        return

    def __len__(self):
        lengths = [len(b) for b in self.buffers.values()]
        return sum(lengths)

    def push(self, experience, **kwargs):
        if self.__len__() >= self.capacity: return
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
    
    def clear(self):
        [buffer.clear() for buffer in self.buffers.values()]



class ReplayBuffer(rllib.buffer.ReplayBuffer):
    # def push_new(self, experience, **kwargs):
    #     breakpoint()
    #     rate = 0.2
    #     index = self.size % self.capacity
    #     if self.size >= self.capacity:
    #         index = (index % int((1-rate)*self.capacity)) + rate*self.capacity
    #         index = int(index)

    #     self.memory[index] = experience
    #     self.size += 1
    def __init__(self, config, capacity, batch_size, device):
        super().__init__(config, capacity, batch_size, device)
        self.raw_horizon = config.raw_horizon
        self.horizon = config.horizon
    def push(self, experience, **kwargs):
        experience.state = sample_state(experience.state, self.raw_horizon, self.horizon)
        experience.next_state = sample_state(experience.next_state, self.raw_horizon, self.horizon)
        self.memory[self.size % self.capacity] = experience
        self.size += 1

    def clear(self):
        del self.memory
        self.memory = np.empty(self.capacity, dtype=Experience)
        self.size = 0



def sample_state(state: rllib.basic.Data, raw_horizon, horizon) :
    state_ = copy.deepcopy(state)
    #hrz30 -> 10
    horizon = 15
    # raw_horizon = 30
    interval = int(raw_horizon / horizon)
    state_.ego = torch.cat((state_.ego[:,interval-1:-1:interval,:], state_.ego[:,-1:,:]), 1)  
    state_.ego_mask = torch.cat((state_.ego_mask[:,interval-1:-1:interval], state_.ego_mask[:,-1:]), 1) 

    state_.obs = torch.cat((state_.obs[:,:,interval-1:-1:interval,:], state_.obs[:,:,-1:,:]), 2)  
    state_.obs_mask = torch.cat((state_.obs_mask[:,:,interval-1:-1:interval], state_.obs_mask[:,:,-1:]), 2)  
    # state_.obs_character = state_.obs_character[:,:,-horizon:,-1]
    return state_


if __name__ == '__main__':
    replay_buffer = ReplayBuffer(None, 101, 2, device='cpu')
    
    import pdb; pdb.set_trace()

    for i in range(206):
        replay_buffer.push(i)


    import pdb; pdb.set_trace()
