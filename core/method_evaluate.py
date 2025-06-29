from turtle import forward
import rllib
import numpy as np
import matplotlib
import torch
import copy



class Evaluate(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name
        if method_name == 'TD3':
            from rllib import td3
            self.critic = config.get('net_critic', td3.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', td3.Actor)(config).to(self.device)
            self.models_to_load = [self.critic, self.actor]
            self.select_action = self.select_action_td3
        elif method_name == 'SAC':
            from rllib import sac
            self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
            self.models_to_load = [self.critic, self.actor]
            self.select_action = self.select_action_sac

        elif method_name == 'IndependentSAC_v0'.upper():
            from rllib import sac
            self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
            # self.models_to_load = [self.critic, self.actor]
            self.models_to_load = [self.actor]
            self.select_actions = self.select_action_isac
            self.select_action = self.select_action_sac

        else:
            raise NotImplementedError('No such method: ' + str(method_name))
        return



    @torch.no_grad()
    def select_action_td3(self, state):
        self.select_action_start()
        state = state.to(self.device)
        action = self.actor(state)


        # print('action: ', action)

        return action


    @torch.no_grad()
    def select_action_sac(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action



    @torch.no_grad()
    def select_action_isac(self, state):
        self.select_action_start()

        ### v1
        # states = rllib.buffer.stack_data(state).cat(dim=0)

        ### v2
        from .model_vectornet import ReplayBufferMultiAgentWithCharacters
        states = rllib.buffer.stack_data(state)
        ReplayBufferMultiAgentWithCharacters.pad_state(None, states)
        states = states.cat(dim=0)

        states = states.to(self.device)

        actions, logprob, mean = self.actor.sample(states)

        return actions


    def store(self, experience, **kwargs):
        return








class EvaluateIndependentSAC(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name

        from rllib import sac
        self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
        self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
        self.models_to_load = [self.critic, self.actor]
        self.buffer_len = 0
        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action.cpu()



    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()

        ### v1
        # states = rllib.buffer.stack_data(state).cat(dim=0)

        ### v2
        from .model_vectornet import ReplayBufferMultiAgentWithCharacters
        states = rllib.buffer.stack_data(state)
        ReplayBufferMultiAgentWithCharacters.pad_state(None, states)
        states = states.cat(dim=0)

        states = states.to(self.device)

        actions, logprob, mean = self.actor.sample(states)

        return actions.cpu()


    def store(self, experience, **kwargs):
        self.buffer_len += 1
        return
    def _get_buffer_len(self):
        return self.buffer_len





class EvaluateIndependentSACMean(EvaluateIndependentSAC):

    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return mean.cpu()



    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()

        ### v1
        # states = rllib.buffer.stack_data(state).cat(dim=0)

        ### v2
        from .model_vectornet import ReplayBufferMultiAgentWithCharacters
        states = rllib.buffer.stack_data(state)
        ReplayBufferMultiAgentWithCharacters.pad_state(None, states)
        states = states.cat(dim=0)

        states = states.to(self.device)

        actions, logprob, mean = self.actor.sample(states)

        return mean.cpu()




class EvaluateSAC(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name

        from rllib import sac
        self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
        self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
        self.models_to_load = [self.critic, self.actor]

        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action.cpu()



    def store(self, experience, **kwargs):
        return



class EvaluatePPO(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name

        from rllib import ppo
        self.policy = config.get('net_ac', ppo.ActorCriticContinuous)(config).to(self.device)

        self.models_to_load = [self.policy]

        return



    def store(self, experience, **kwargs):
        return

    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        
        state = state.to(self.device)
        action, mean, dist = self.policy(state)

        # action_logprobs, state_value, dist_entropy = self.policy.evaluate(state, action)
        print('\n')
        print('action: ', action.cpu().data, 'mean: ', mean.cpu().data, 'value: ', 'variance:', dist.variance)
        # import pdb; pdb.set_trace()
        return mean.cpu().data



class EvaluateSACAdv(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name

        from rllib import sac
        self.critic = config.get('net_critic', sac.Critic)(config, model_id=0).to(self.device)
        self.actor = config.get('net_actor', sac.Actor)(config, model_id=0).to(self.device)

        config_adv = copy.copy(config)
        config_adv.set('dim_action', 1)
        self.critic_adv = config.get('net_critic', sac.Critic)(config_adv, model_id=1).to(self.device)
        self.actor_adv = config.get('net_actor', sac.Actor)(config_adv, model_id=1).to(self.device)

        self.models_to_load = [self.critic, self.actor, self.critic_adv, self.actor_adv]

        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)

        action, logprob, mean = self.actor.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action.cpu()


    @torch.no_grad()
    def select_action_adv(self, state):
        state = state.to(self.device)

        action, logprob, mean = self.actor_adv.sample(state)

        # print('action: ', action, mean)
        # return mean
        return action.cpu()




    def store(self, experience, **kwargs):
        return





class EvaluateSACAdvDecouple(EvaluateSACAdv):

    def select_method(self):
        config, method_name = self.config, self.method_name
        config_adv = config.config_adv

        config_adv.set('evaluate', config.evaluate)
        config_adv.set('device', config.device)
        config_adv.set('method_name', config.method_name)
        config_adv.set('net_actor_fe', config.net_actor_fe)
        config_adv.set('net_critic_fe', config.net_critic_fe)
        config_adv.set('dim_state', config.dim_state)
        config_adv.set('dim_action', 1)


        from rllib import sac
        self.critic = config.get('net_critic', sac.Critic)(config, model_id=0).to(self.device)
        self.actor = config.get('net_actor', sac.Actor)(config, model_id=0).to(self.device)

        self.critic_adv = config.get('net_critic', sac.Critic)(config_adv, model_id=1).to(self.device)
        self.actor_adv = config.get('net_actor', sac.Actor)(config_adv, model_id=1).to(self.device)

        self.models_to_load = [self.critic, self.actor, self.critic_adv, self.actor_adv]

        return



class EvaluateSACRecog(rllib.EvaluateSingleAgent):
    character_loss = torch.nn.MSELoss()
    def select_method(self):
        config, method_name = self.config, self.method_name
        from core.method_isac_recog import Actor
        # class Actor(Actor):
        #     def forward(self, state):
        #         #add character into state
        #         obs_character = self.recog(state)
        #         # obs_character = np.random.uniform(0,1, size=obs_character)
        #         # obs_character = torch.full(obs_character.size(), 0.880797).to(self.device)
        #         # print(obs_character)
        #         #####
        #         x = self.fe(state, obs_character)
        #         mean = self.mean_no(self.mean(x))
        #         logstd = self.std_no(self.std(x))
        #         logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        #         return mean, logstd *0.5
        # self.critic = config.get('net_critic', Critic)(config).to(self.device)
        self.actor = config.get('net_actor', Actor)(config).to(self.device)
        # self.models_to_load = [self.critic, self.actor]
        self.models_to_load = [self.actor]
        return
    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)
        action, logprob, mean = self.actor.sample(state)
        # print('action: ', action, mean)
        # return mean
        return action.cpu()

    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()
        from .model_vectornet import ReplayBufferMultiAgentWithCharacters
        states = rllib.buffer.stack_data(state)
        ReplayBufferMultiAgentWithCharacters.pad_state(None, states)
        states = states.cat(dim=0)

        states = states.to(self.device)

        actions, logprob, mean = self.actor.sample(states)

        #计算
        recog_charater = self.actor.fe.get_recog_obs_svos()
        real_character = states.obs_character[:,:,-1]
        recog_charater = torch.where(real_character == np.inf, torch.tensor(np.inf, dtype=torch.float32, device=states.obs.device),recog_charater)
        real_character = real_character[~torch.isinf(real_character)]
        recog_charater = recog_charater[~torch.isinf(recog_charater)]
        # character_loss = self.character_loss(recog_charater, real_character)
        dev = torch.abs(recog_charater-real_character)
        mean_dev = torch.mean(dev)
        std_dev = torch.std(dev)
        # RMSE_loss = torch.sqrt(character_loss)
        return actions.cpu(), mean_dev, std_dev

    def store(self, experience, **kwargs):
        return



class EvaluateSACRecogWoattn(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name
        from core.method_wo_attention import Actor
        # class Actor(Actor):
        #     def forward(self, state):
        #         #add character into state
        #         obs_character = self.recog(state)
        #         # obs_character = np.random.uniform(0,1, size=obs_character)
        #         # obs_character = torch.full(obs_character.size(), 0.880797).to(self.device)
        #         # print(obs_character)
        #         #####
        #         x = self.fe(state, obs_character)
        #         mean = self.mean_no(self.mean(x))
        #         logstd = self.std_no(self.std(x))
        #         logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        #         return mean, logstd *0.5
        # self.critic = config.get('net_critic', Critic)(config).to(self.device)
        self.actor = config.get('net_actor', Actor)(config).to(self.device)
        # self.models_to_load = [self.critic, self.actor]
        # breakpoint()
        self.models_to_load = [self.actor]
        return
    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)
        action, logprob, mean = self.actor.sample(state)
        # print('action: ', action, mean)
        # return mean
        return action.cpu()
    def store(self, experience, **kwargs):
        return


class EvaluateRecogV1(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name
        from core.method_recog_new_action import Actor
        self.actor = config.get('net_actor', Actor)(config).to(self.device)
        self.models_to_load = [self.actor]
        return
    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        action, _, _ = self.actor.sample(state.to(self.device))
        action = action.cpu()
        # print('select_action', action)
        return action

    def store(self, experience, **kwargs):
        return

class EvaluateRecogV2(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name
        from core.method_recog_action_dynamic import Actor
        self.actor = config.get('net_actor', Actor)(config).to(self.device)
        self.models_to_load = [self.actor]
        return
    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        action, _, _ = self.actor.sample(state.to(self.device))
        action = action.cpu()
        # print('select_action', action.shape)
        # valid_len = state.obs_character.shape[1]
        # action = torch.Tensor(1,1,1).uniform_(0.1,0.9)
        # action = action.repeat(1,19,1)
        # action[0,valid_len:] = -1
        return action

    def store(self, experience, **kwargs):
        return

    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()
        from .model_vectornet import ReplayBufferMultiAgentWithCharacters
        states = rllib.buffer.stack_data(state)
        ReplayBufferMultiAgentWithCharacters.pad_state(None, states)
        states = states.cat(dim=0)

        states = states.to(self.device)

        actions, logprob, mean = self.actor.sample(states)

        return actions.cpu()

class EvaluateSupervise(rllib.EvaluateSingleAgent):
    def select_method(self):
        config, method_name = self.config, self.method_name

        from core.method_supervise_close_loop import Actor
        self.actor = config.get('net_actor', Actor)(config).to(self.device)
        self.models_to_load = [self.actor]
        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        state = state.to(self.device)

        # print(state.ego[:,:10,:])
        action, logprob, mean = self.actor.sample(state)

        return action.cpu()

    @torch.no_grad()
    def select_actions(self, state):
        self.select_action_start()
        from .model_vectornet import ReplayBufferMultiAgentWithCharacters
        states = rllib.buffer.stack_data(state)
        ReplayBufferMultiAgentWithCharacters.pad_state(None, states)
        states = states.cat(dim=0)

        states = states.to(self.device)

        actions, logprob, mean = self.actor.sample(states)

        #计算
        recog_charater = self.actor.fe.get_recog_obs_svos()
        real_character = states.obs_character[:,:,-1]
        recog_charater = torch.where(real_character == np.inf, torch.tensor(np.inf, dtype=torch.float32, device=states.obs.device),recog_charater)
        real_character = real_character[~torch.isinf(real_character)]
        recog_charater = recog_charater[~torch.isinf(recog_charater)]
        # character_loss = self.character_loss(recog_charater, real_character)
        dev = recog_charater-real_character
        mean_dev = torch.mean(torch.abs(dev))
        std_dev = torch.std(dev)
        # RMSE_loss = torch.sqrt(character_loss)
        # RMSE_loss = torch.sqrt(character_loss)
        # if torch.isinf(dev).any() :breakpoint()
        # if torch.isnan(dev).any() :breakpoint()
        # if torch.isinf(mean_dev).any() :breakpoint()
        # if torch.isnan(std_dev).any() :breakpoint()
        return actions.cpu(), mean_dev, std_dev


    def store(self, experience, **kwargs):
        return