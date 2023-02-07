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


class EnvInteractiveMultiAgentFixSvo(EnvInteractiveMultiAgent):
    def __init__(self, config: rllib.basic.YamlConfig, writer: rllib.basic.Writer, env_index=0, ego_svo=0.5,
    other_svo=0.6):
        super().__init__(config, writer, env_index)
        self.ego_svo = ego_svo
        self.other_svo = other_svo

    def reset(self):
        self.step_reset += 1
        self.time_step = 0

        self.dataset = self.datasets[self.step_reset % self.num_cases]
        self.num_steps = len(self.dataset)

        self.scenario = self.dataset.scenario
        self.agents_master = self.dataset.agents_master
        self.agents_master.destroy()

        self.scenario.reset(self.ego_svo, self.other_svo, self.step_reset, sa=self.sa)
        self.num_vehicles = self.scenario.num_vehicles
        self.num_vehicles_max = self.scenario.num_vehicles_max
        self.num_agents = self.scenario.num_agents
        self.agents_master.reset(self.num_steps)

        self.scenario.register_agents(self.agents_master)

        self.state = self.agents_master.observe(self.step_reset, self.time_step)

        self.metric.on_episode_start(self.step_reset, self.scenario, self.agents_master, self.num_steps, self.num_agents)
        self.recorder.record_scenario(self.step_reset, self.scenario, self.agents_master)
        return self.state[self.slice()]