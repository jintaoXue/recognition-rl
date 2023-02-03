from universe import EnvInteractiveMultiAgent 
import copy
import rllib
import torch

class EnvInteractiveMultiAgentActSvo(EnvInteractiveMultiAgent):

    def step(self, action):
        self.time_step += 1

        episode_info = self.check_episode()
        reward = self.reward_func.run_step(self.state, action, self.agents_master, episode_info)
        done = episode_info.done
        ### metric step
        self.metric.on_episode_step(self.time_step, reward, episode_info)
        self.recorder.record_vehicles(self.time_step, self.agents_master, episode_info)

        ### step
        self.agents_master.run_step(action, self.state)

        ### reset done agents, only for multi-agent
        self.reset_done_agents(episode_info)

        ### next_state
        next_state = self.agents_master.observe(self.step_reset, self.time_step)
        state = self.state
        self.state = copy.copy(next_state)

        ### pad next_state, only for multi-agent
        next_state_vis = [s.vi for s in next_state]
        for s in state:
            if s.vi not in next_state_vis:
                next_state.append(s)
        next_state = sorted(next_state, key=lambda x: x.vi)

        ### experience
        experience = []
        for s, a, ns, r, d in zip(state, action, next_state, reward, episode_info.finish):
            e = rllib.basic.Data(vi=s.vi,
                state=s.to_tensor().unsqueeze(0),
                action=torch.from_numpy(a).unsqueeze(0),
                next_state=ns.to_tensor().unsqueeze(0),
                reward=torch.tensor([r], dtype=torch.float32),
                done=torch.tensor([d], dtype=torch.float32),
            )
            experience.append(e)

        ### metric end
        if done:
            self.metric.on_episode_end()
            self.recorder.save()
        return experience[self.slice()], done, episode_info.to_dict()