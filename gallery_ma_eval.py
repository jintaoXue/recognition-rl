import rllib
import universe
import ray

import os
import copy
import torch




def init(config, mode, Env) -> universe.EnvMaster_v1:
    # repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    # config.set('github_repos', repos)

    from core.method_isac_v0 import IndependentSAC_v0 as Method

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from core.method_evaluate import EvaluateIndependentSAC as Method

    from universe import EnvMaster_v1 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env, method_cls=Method)
    return env_master


def init_recog(config, mode, Env, Method) -> universe.EnvMaster_v1:
    # repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    # config.set('github_repos', repos)
    
    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    

    from universe import EnvMaster_v1 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env, method_cls=Method)
    return env_master

def init_fix_svo(config, mode, Env, Method, svo):

    # repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    # config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from universe import EnvMaster_v2 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env, method_cls=Method, ego_svo= svo, other_svo=svo)
    return env_master





################################################################################################
##### evaluate, training setting ###############################################################
################################################################################################


# def evaluate_ray_isac_adaptive_character__bottleneck(config, mode='evaluate', scale=5):
#     from universe import EnvInteractiveMultiAgent as Env

#     ### env param
#     from config.bottleneck_evaluate import config_env__with_character
#     config.set('envs', [config_env__with_character] *scale)

#     ### method param
#     from config.method import config_isac__adaptive_character as config_method
#     config.set('methods', [config_method])

#     return init(config, mode, Env)



def evaluate_ray_isac_adaptive_character__intersection(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.intersection_evaluate import config_env__with_character
    config_env__with_character.set('scenario_name', 'intersection_v2')
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



# def evaluate_ray_isac_adaptive_character__merge(config, mode='evaluate', scale=5):
#     from universe import EnvInteractiveMultiAgent as Env

#     ### env param
#     from config.merge_evaluate import config_env__with_character
#     config.set('envs', [config_env__with_character] *scale)

#     ### method param
#     from config.method import config_isac__adaptive_character as config_method
#     config.set('methods', [config_method])

#     return init(config, mode, Env)



def evaluate_ray_isac_adaptive_character__roundabout(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.roundabout_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



################################################################################################
##### evaluate, training setting robust ########################################################
################################################################################################



def evaluate_ray_isac_robust_character__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.bottleneck_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_robust_character__intersection(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.intersection_evaluate import config_env__with_character
    config_env__with_character.set('scenario_name', 'intersection_v2')
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_robust_character__merge(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.merge_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_robust_character__roundabout(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.roundabout_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)




################################################################################################
##### evaluate, training setting copo ##########################################################
################################################################################################





from gallery_sa import get_sac__bottleneck__robust_character_config
from gallery_sa import get_sac__intersection__robust_character_config
from gallery_sa import get_sac__merge__robust_character_config
from gallery_sa import get_sac__roundabout__robust_character_config




def evaluate_ray_isac_robust_character_copo__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
    from utils.agents_master_copo import AgentListMasterTuneSVO as agents_master_cls
    from utils.reward import RewardFunctionGlobalCoordination as reward_func

    from config.bottleneck_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', reward_func)
    config_env__with_character.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_robust_character_copo__intersection(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
    from utils.agents_master_copo import AgentListMasterTuneSVO as agents_master_cls
    from utils.reward import RewardFunctionGlobalCoordination as reward_func

    from config.intersection_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', reward_func)
    config_env__with_character.set('config_neural_policy', get_sac__intersection__robust_character_config(config))
    config_env__with_character.set('scenario_name', 'intersection_v2')

    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_robust_character_copo__merge(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
    from utils.agents_master_copo import AgentListMasterTuneSVO as agents_master_cls
    from utils.reward import RewardFunctionGlobalCoordination as reward_func

    from config.merge_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', reward_func)
    config_env__with_character.set('config_neural_policy', get_sac__merge__robust_character_config(config))

    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_robust_character_copo__roundabout(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
    from utils.agents_master_copo import AgentListMasterTuneSVO as agents_master_cls
    from utils.reward import RewardFunctionGlobalCoordination as reward_func

    from config.roundabout_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', reward_func)
    config_env__with_character.set('config_neural_policy', get_sac__roundabout__robust_character_config(config))

    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)









################################################################################################
##### evaluate, training setting no ############################################################
################################################################################################



def evaluate_ray_isac_no_character__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.bottleneck_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_no_character__intersection(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.intersection_evaluate import config_env__with_character
    config_env__with_character.set('scenario_name', 'intersection_v2')
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_no_character__merge(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.merge_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_no_character__roundabout(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.roundabout_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)








################################################################################################
##### evaluate, training setting copo adv ######################################################
################################################################################################




def evaluate_ray_isac_robust_character_copo_adv__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
    from utils.agents_master_copo import AgentListMasterTuneSVOAdv as agents_master_cls
    from utils.reward import RewardFunctionGlobalCoordination as reward_func

    from config.bottleneck_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', reward_func)
    config_env__with_character.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_robust_character_copo_adv__intersection(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
    from utils.agents_master_copo import AgentListMasterTuneSVOAdv as agents_master_cls
    from utils.reward import RewardFunctionGlobalCoordination as reward_func

    from config.intersection_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', reward_func)
    config_env__with_character.set('config_neural_policy', get_sac__intersection__robust_character_config(config))
    config_env__with_character.set('scenario_name', 'intersection_v2')

    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_robust_character_copo_adv__merge(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
    from utils.agents_master_copo import AgentListMasterTuneSVOAdv as agents_master_cls
    from utils.reward import RewardFunctionGlobalCoordination as reward_func

    from config.merge_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', reward_func)
    config_env__with_character.set('config_neural_policy', get_sac__merge__robust_character_config(config))

    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_robust_character_copo_adv__roundabout(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
    from utils.agents_master_copo import AgentListMasterTuneSVOAdv as agents_master_cls
    from utils.reward import RewardFunctionGlobalCoordination as reward_func

    from config.roundabout_evaluate import config_env__with_character
    config_env__with_character.set('neural_vehicle_cls', neural_vehicle_cls)
    config_env__with_character.set('agents_master_cls', agents_master_cls)
    config_env__with_character.set('reward_func', reward_func)
    config_env__with_character.set('config_neural_policy', get_sac__roundabout__robust_character_config(config))

    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)






















################################################################################################
##### evaluate, idm ############################################################################
################################################################################################




def evaluate_ray_isac_idm__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.bottleneck_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_idm__intersection(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.intersection_evaluate import config_env__with_character
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
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)



def evaluate_ray_isac_idm__roundabout(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.roundabout_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)

















################################################################################################
##### evaluate, assign character, adaptive #####################################################
################################################################################################



def evaluate_ray_isac_adaptive_character_assign__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.bottleneck_evaluate import config_env__with_character_assign
    config.set('envs', [config_env__with_character_assign] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_adaptive_character_assign__intersection(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.intersection_evaluate import config_env__with_character_assign
    config_env__with_character_assign.set('scenario_name', 'intersection_v2')
    config.set('envs', [config_env__with_character_assign] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_adaptive_character_assign__merge(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.merge_evaluate import config_env__with_character_assign
    config.set('envs', [config_env__with_character_assign] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_adaptive_character_assign__roundabout(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.roundabout_evaluate import config_env__with_character_assign
    config.set('envs', [config_env__with_character_assign] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)




################################################################################################
##### evaluate, assign character, robust #######################################################
################################################################################################



def evaluate_ray_isac_robust_character_assign__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.bottleneck_evaluate import config_env__with_character_assign
    config.set('envs', [config_env__with_character_assign] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_robust_character_assign__intersection(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.intersection_evaluate import config_env__with_character_assign
    config_env__with_character_assign.set('scenario_name', 'intersection_v2')
    config.set('envs', [config_env__with_character_assign] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_robust_character_assign__merge(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.merge_evaluate import config_env__with_character_assign
    config.set('envs', [config_env__with_character_assign] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_robust_character_assign__roundabout(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env

    ### env param
    from config.roundabout_evaluate import config_env__with_character_assign
    config.set('envs', [config_env__with_character_assign] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)






################################################################################################
##### evaluate, diversity ######################################################################
################################################################################################



def evaluate_ray_isac_adaptive_character_diversity__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env
    from core.method_evaluate import EvaluateIndependentSACMean as Method
    # from core.method_evaluate import EvaluateIndependentSAC as Method

    config.set('num_episodes', 11)
    ### env param
    from config.bottleneck_evaluate import config_env__with_character

    from utils.scenarios_bottleneck import ScenarioBottleneckEvaluate_fix_others as scenario_cls
    config_env__with_character.set('scenario_cls', scenario_cls)

    config_env__with_character.set('num_vehicles_range', rllib.basic.BaseData(min=20, max=20))
    config_env__with_character.set('recorder_cls', universe.Recorder)
    config_env__with_character.set('randomization_index', 15)
    config.description += f'--data-{config_env__with_character.randomization_index}'
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    ### ! todo: set method
    env_master = init(config, mode, Env)
    for config_method in env_master.config_methods:
        config_method.set('method_cls', Method)
    return env_master










################################################################################################
##### evaluate, social behaviour ###############################################################
################################################################################################


def evaluate_ray_isac_adaptive_character__social_behavior__bottleneck(config, mode='evaluate', scale=5):
    from core.env_eval import EnvInteractiveMultiAgent_v1 as Env

    ### env param
    from config.bottleneck_evaluate import config_env__with_character
    config_env__with_character.set('num_steps', 200)
    config_env__with_character.set('num_vehicles_range', rllib.basic.BaseData(min=10, max=10))
    config_env__with_character.set('recorder_cls', universe.Recorder)
    config_env__with_character.set('spawn_interval', 2)
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_adaptive_character__social_behavior__intersection(config, mode='evaluate', scale=5):
    from core.env_eval import EnvInteractiveMultiAgent_v1 as Env

    ### env param
    from config.intersection_evaluate import config_env__with_character
    config_env__with_character.set('num_steps', 200)
    config_env__with_character.set('num_vehicles_range', rllib.basic.BaseData(min=10, max=10))
    config_env__with_character.set('recorder_cls', universe.Recorder)
    config_env__with_character.set('spawn_interval', 1)
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_adaptive_character__social_behavior__merge(config, mode='evaluate', scale=5):
    from core.env_eval import EnvInteractiveMultiAgent_v1 as Env

    ### env param
    from config.merge_evaluate import config_env__with_character
    config_env__with_character.set('num_steps', 200)
    config_env__with_character.set('num_vehicles_range', rllib.basic.BaseData(min=10, max=10))
    config_env__with_character.set('recorder_cls', universe.Recorder)
    config_env__with_character.set('spawn_interval', 2)
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)


def evaluate_ray_isac_adaptive_character__social_behavior__roundabout(config, mode='evaluate', scale=5):
    from core.env_eval import EnvInteractiveMultiAgent_v1 as Env

    ### env param
    from config.roundabout_evaluate import config_env__with_character
    config_env__with_character.set('num_steps', 200)
    config_env__with_character.set('num_vehicles_range', rllib.basic.BaseData(min=10, max=10))
    config_env__with_character.set('recorder_cls', universe.Recorder)
    config_env__with_character.set('spawn_interval', 1)
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env)
#################################################################################
##############evaluate recog bottleneck##########################################
#################################################################################

def evaluate_ray_isac_adaptive_character__bottleneck(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env
    from core.method_evaluate import EvaluateIndependentSAC as Method

    ### env param
    from config.bottleneck_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)

def evaluate_ray_isac_adaptive_character__bottleneck_fix_svo(config, svo, mode='evaluate', scale=1):
    from utils.env import EnvInteractiveMultiAgentFixSvo as Env
    from core.method_evaluate import EvaluateIndependentSAC as Method

    ### env param
    from config.bottleneck_evaluate import config_env__fix_svo
    config.set('envs', [config_env__fix_svo] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init_fix_svo(config, mode, Env, Method, svo)

def evaluate_ray_RILMthM__bottleneck(config, mode='train', scale=5):
    from universe import EnvInteractiveMultiAgent as Env
    from core.method_evaluate import EvaluateSACRecog as Method
    
    ### env param
    from config.bottleneck_evaluate import config_env__with_character as config_bottleneck
    # config_bottleneck.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))

    config.set('envs', [
        config_bottleneck,
    ] *scale)

    ### method param
    from config.method import config_recog_multi_agent as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)


def evaluate_ray_RILMthM__bottleneck_assign_case(config, mode='train', scale=5):
    from universe import EnvInteractiveMultiAgent as Env
    from core.method_evaluate import EvaluateSACRecog as Method
    # config.set('num_episodes', 1)
    ### env param
    from config.bottleneck_evaluate import config_env__fix_svo as config_bottleneck
    # config_bottleneck.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))
    config_bottleneck.set('randomization_index', 11)
    config.set('envs', [
        config_bottleneck,
    ] *scale)

    ### method param
    from config.method import config_recog_multi_agent as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)

def evaluate_ray_RILEnvM__bottleneck(config, mode='train', scale=5):
    from utils.env import EnvInteractiveMultiAgentActSvo as Env
    #todo
    from core.method_evaluate import EvaluateRecogV2 as Method
    
    ### env param
    from config.bottleneck_evaluate import config_env__actsvo_multiagent as config_bottleneck
    from gallery_ma import get_sac__bottleneck__new_action_config
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__new_action_config(config))

    config.set('envs', [
        config_bottleneck,
    ] *scale)

    ### method param
    from config.method import config_action_svo_multiagent as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)


def evalute_IL__multiagent__bottleneck(config, mode='train', scale=5):
    from universe import EnvInteractiveMultiAgent as Env
    #todo
    from core.method_evaluate import EvaluateSupervise as Method

    ### env param
    from config.bottleneck_evaluate import config_env__with_character as config_bottleneck
    
    config.set('envs', [
        config_bottleneck
    ] * scale)

    ### method param
    from config.method import config_supervise_multi as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)

def evalute_ray_supervise__multiagent__bottleneck(config, mode='train', scale=5):
    from universe import EnvInteractiveMultiAgent as Env
    #todo
    from core.method_evaluate import EvaluateSupervise as Method

    ### env param
    from config.bottleneck_evaluate import config_env__with_character as config_bottleneck
    
    config.set('envs', [
        config_bottleneck
    ]*scale)

    ### method param
    from config.method import config_supervise_multi as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)

def evalute_ray_supervise__multiagent__bottleneck_assign_case(config, mode='train', scale=5):
    from universe import EnvInteractiveMultiAgent as Env
    #todo
    from core.method_evaluate import EvaluateSupervise as Method

    ### env param
    from config.bottleneck_evaluate import config_env__fix_svo as config_bottleneck
    config_bottleneck.set('randomization_index', 11)
    config.set('envs', [
        config_bottleneck
    ]*scale)

    ### method param
    from config.method import config_supervise_multi as config_method
    config_method.set('raw_horizon', 30)
    config_method.set('horizon', 5)
    config.set('methods', [config_method])
    return init_recog(config, mode, Env, Method)

#################################################################################
##############evaluate recog merge###############################################
#################################################################################

def evaluate_ray_isac_adaptive_character__merge(config, mode='evaluate', scale=5):
    from universe import EnvInteractiveMultiAgent as Env
    from core.method_evaluate import EvaluateIndependentSAC as Method

    ### env param
    from config.merge_evaluate import config_env__with_character
    config.set('envs', [config_env__with_character] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)

def evaluate_ray_isac_adaptive_character__merge_fix_svo(config, svo, mode='evaluate', scale=1):
    from utils.env import EnvInteractiveMultiAgentFixSvo as Env
    from core.method_evaluate import EvaluateIndependentSAC as Method

    ### env param
    from config.merge_evaluate import config_env__fix_svo
    config.set('envs', [config_env__fix_svo] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init_fix_svo(config, mode, Env, Method, svo)

def evaluate_ray_RILMthM__merge(config, mode='train', scale=1):
    from universe import EnvInteractiveMultiAgent as Env
    from core.method_evaluate import EvaluateSACRecog as Method
    
    ### env param
    from config.merge_evaluate import config_env__with_character as config_merge
    # config_bottleneck.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))

    config.set('envs', [
        config_merge,
    ] *scale)

    ### method param
    from config.method import config_recog_multi_agent as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)

def evaluate_ray_RILEnvM__merge(config, mode='train', scale=1):
    #########to do ########
    from utils.env import EnvInteractiveMultiAgentActSvo as Env
    #todo
    from core.method_evaluate import EvaluateRecogV2 as Method
    
    ### env param
    from config.bottleneck_evaluate import config_env__actsvo_multiagent as config_bottleneck
    from gallery_ma import get_sac__bottleneck__new_action_config
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__new_action_config(config))

    config.set('envs', [
        config_bottleneck,
    ] *scale)

    ### method param
    from config.method import config_action_svo_multiagent as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)

def evalute_ray_supervise_offline_multiagent__merge(config, mode='train', scale=1):
    from universe import EnvInteractiveMultiAgent as Env
    #todo
    from core.method_evaluate import EvaluateSupervise as Method

    ### env param
    from config.merge_evaluate import config_env__with_character as config_merge
    
    config.set('envs', [
        config_merge,
    ])

    ### method param
    from config.method import config_supervise_multi as config_method
    config.set('methods', [config_method])

    return init_recog(config, mode, Env, Method)


