import rllib
import universe

import os
import copy

import torch





def sac__bottleneck(config, mode):
    repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    config.set('github_repos', repos)

    ### env param
    from config.train import config_bottleneck
    config_env = copy.copy(config_bottleneck)
    Env = universe.EnvInteractiveSingleAgent

    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)
    
    ### method param
    from core.model_vectornet import SAC as Method
    from core.model_vectornet import ReplayBufferSingleAgentWithCharacters as ReplayBuffer
    from core.model_vectornet import PointNetWithAgentHistory as FeatureExtractor

    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_actor_fe', FeatureExtractor)
    config.set('net_critic_fe', FeatureExtractor)
    config.set('buffer', ReplayBuffer)


    model_name = Method.__name__ + '-' + Env.__name__
    writer = rllib.basic.create_dir(config, model_name, mode=mode)

    if mode != 'train':
        from core import method_evaluate_bound as method_evaluate
        Method = method_evaluate.Evaluate
    
    env = Env(config.env, writer)
    config.set('dim_state', env.dim_state)
    config.set('dim_action', env.dim_action)
    method = Method(config, writer)
    return writer, env, method






