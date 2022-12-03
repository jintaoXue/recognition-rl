import rllib
import universe
import ray

import os
import copy
import torch




def init(config, mode, Env) -> universe.EnvMaster_v0:
    repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    config.set('github_repos', repos)

    from core.method_isac_v0 import IndependentSAC_v0 as Method

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from core.method_evaluate import EvaluateIndependentSAC as Method

    from universe import EnvMaster_v0 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env)
    return env_master










################################################################################################
##### evaluate, idm ############################################################################
################################################################################################



from utils.agents_master_normal_bg import IdmVehicleAsNeural as neural_vehicle_cls
from utils.agents_master_normal_bg import AgentListMasterNormalBackground as agents_master_cls


def evaluate_ray_isac_idm__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.bottleneck_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', universe.RewardFunc)
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_idm__intersection(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.intersection_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', universe.RewardFunc)
    config_env__with_character.set('scenario_name', 'intersection_v2')
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_idm__merge(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.merge_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', universe.RewardFunc)
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_idm__roundabout(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.roundabout_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', universe.RewardFunc)
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)












