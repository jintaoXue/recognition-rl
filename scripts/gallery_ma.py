import rllib
import universe

import os
import copy
from typing import Tuple

import torch




def init(config, mode, Env, Method) -> Tuple[rllib.basic.Writer, universe.EnvMaster, rllib.template.Method]:
    repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer = rllib.basic.create_dir(config, model_name, mode=mode)

    if mode != 'train':
        from core import method_evaluate
        Method = method_evaluate.Evaluate
    
    env = Env(config.env, writer)
    config.set('dim_state', env.dim_state)
    config.set('dim_action', env.dim_action)
    method = Method(config, writer)
    return writer, env, method




def isac_no_character__bottleneck(config, mode):
    ### env param
    from config.train import config_bottleneck
    config_env = copy.copy(config_bottleneck)
    Env = universe.EnvInteractiveMultiAgent

    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)
    
    ### method param
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.model_vectornet import ReplayBufferMultiAgentWithCharacters as ReplayBuffer
    from core.model_vectornet import PointNetWithAgentHistory as FeatureExtractor

    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_actor_fe', FeatureExtractor)
    config.set('net_critic_fe', FeatureExtractor)
    config.set('buffer', ReplayBuffer)

    writer, env, method = init(config, mode, Env, Method)
    return writer, env, method




def isac_adaptive_character__bottleneck(config, mode='train'):
    ### env param
    from config.train import config_bottleneck__with_character
    config_env = copy.copy(config_bottleneck__with_character)
    Env = universe.EnvInteractiveMultiAgent

    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)

    ### method param
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.model_vectornet import ReplayBufferMultiAgentWithCharacters as ReplayBuffer
    from core.model_vectornet import PointNetWithCharactersAgentHistory as FeatureExtractor

    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_actor_fe', FeatureExtractor)
    config.set('net_critic_fe', FeatureExtractor)
    config.set('buffer', ReplayBuffer)

    writer, env, method = init(config, mode, Env, Method)
    return writer, env, method








def isac__intersection(config, mode):
    ### env param
    from config.train import config_intersection
    config_env = copy.copy(config_intersection)
    Env = universe.EnvInteractiveMultiAgent

    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)
    
    ### method param
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.model_vectornet import ReplayBufferMultiAgentWithCharacters as ReplayBuffer
    from core.model_vectornet import PointNetWithAgentHistory as FeatureExtractor

    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_actor_fe', FeatureExtractor)
    config.set('net_critic_fe', FeatureExtractor)
    config.set('buffer', ReplayBuffer)

    writer, env, method = init(config, mode, Env, Method)
    return writer, env, method























def evaluate_isac_adaptive_character__bottleneck(config, mode='train'):
    ### env param
    from config.evaluate import config_bottleneck__with_character
    config_env = copy.copy(config_bottleneck__with_character)
    Env = universe.EnvInteractiveMultiAgent

    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)

    ### method param
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.model_vectornet import ReplayBufferMultiAgentWithCharacters as ReplayBuffer
    from core.model_vectornet import PointNetWithCharactersAgentHistory as FeatureExtractor

    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_actor_fe', FeatureExtractor)
    config.set('net_critic_fe', FeatureExtractor)
    config.set('buffer', ReplayBuffer)

    writer, env, method = init(config, mode, Env, Method)
    return writer, env, method


























































def debug__isac_adaptive_character__merge(config, mode='train'):
    ### env param
    from config.train import config_merge__with_character
    config_env = copy.copy(config_merge__with_character)
    Env = universe.EnvInteractiveMultiAgent

    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)

    ### method param
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.model_vectornet import ReplayBufferMultiAgentWithCharacters as ReplayBuffer
    from core.model_vectornet import PointNetWithCharactersAgentHistory as FeatureExtractor

    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_actor_fe', FeatureExtractor)
    config.set('net_critic_fe', FeatureExtractor)
    config.set('buffer', ReplayBuffer)

    writer, env, method = init(config, mode, Env, Method)
    return writer, env, method



def debug__isac_adaptive_character__roundabout(config, mode='train', scale=5):
    ### env param
    from config.train import config_roundabout__with_character
    config_env = copy.copy(config_roundabout__with_character)
    Env = universe.EnvInteractiveMultiAgent

    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)

    ### method param
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.model_vectornet import ReplayBufferMultiAgentWithCharacters as ReplayBuffer
    from core.model_vectornet import PointNetWithCharactersAgentHistory as FeatureExtractor

    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_actor_fe', FeatureExtractor)
    config.set('net_critic_fe', FeatureExtractor)
    config.set('buffer', ReplayBuffer)

    writer, env, method = init(config, mode, Env, Method)
    return writer, env, method



def debug__isac_adaptive_character__intersection(config, mode='train', scale=5):
    ### env param
    from config.train import config_intersection__with_character
    config_env = copy.copy(config_intersection__with_character)
    Env = universe.EnvInteractiveMultiAgent

    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)

    ### method param
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.model_vectornet import ReplayBufferMultiAgentWithCharacters as ReplayBuffer
    from core.model_vectornet import PointNetWithCharactersAgentHistory as FeatureExtractor

    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_actor_fe', FeatureExtractor)
    config.set('net_critic_fe', FeatureExtractor)
    config.set('buffer', ReplayBuffer)

    writer, env, method = init(config, mode, Env, Method)
    return writer, env, method


