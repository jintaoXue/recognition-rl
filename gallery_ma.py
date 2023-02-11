import rllib
import universe
import ray

import os
import copy
from typing import Tuple

import torch


# from universe import EnvInteractiveMultiAgent as Env
# from core.method_isac_v0 import IndependentSAC_v0 as Method



def init(config, mode, Env, Method) -> Tuple[rllib.basic.Writer, universe.EnvMaster_v0, rllib.template.Method]:
    repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from universe import EnvMaster_v0 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env)

    config_method = env_master.config_methods[0]
    Method = ray.remote(num_cpus=config_method.num_cpus, num_gpus=config_method.num_gpus)(Method).remote
    method = Method(config_method, writer)
    method.reset_writer.remote()

    return writer, env_master, method


############################################################################
#### model #################################################################
############################################################################

def get_sac__new_bottleneck__adaptive_character_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.model_vectornet import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    from core.model_vectornet import PointNetWithCharactersAgentHistoryCutstate as FeatureExtractor
    # from core.recognition_net import PointNetNewAction as FeatureExtractor
    model_dir = config.action_policy_model_dir
    model_num = config.action_policy_model_num
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,

        # model_dir='~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method',
        # model_dir='~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method',
        
        # model_dir = '~/github/zdk/recognition-rl/models/origin_no_history_bottleneck/',
        model_dir = model_dir,
        # model_num=865800,
        # model_num=445600,
        model_num = model_num,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy

def get_sac__bottleneck__new_action_config(config):
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    from core.model_vectornet import ReplayBufferMultiAgentMultiWorker as ReplayBuffer
    # from core.model_vectornet import PointNetWithCharactersAgentHistory as FeatureExtractor
    from core.recognition_net import PointNetwithActionSVO as FeatureExtractor
    model_dir = config.action_policy_model_dir
    model_num = config.action_policy_model_num
    config_neural_policy = rllib.basic.YamlConfig(
        evaluate=config.evaluate,
        method_name=Method.__name__,
        # model_dir='~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method',
        # model_dir = '~/github/zdk/recognition-rl/models/origin_no_history_bottleneck/',
        # # model_num=865800,
        # model_num=445600,
        model_dir = model_dir,
        model_num = model_num,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        net_actor_fe=FeatureExtractor,
        net_critic_fe=FeatureExtractor,
        buffer=ReplayBuffer,
    )
    return config_neural_policy
############################################################################
#### bottleneck ############################################################
############################################################################


def ray_isac_no_character__bottleneck(config, mode='train', scale=10):
    ### env param
    from config.bottleneck import config_env
    config.set('envs', [config_env] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)



def ray_isac_robust_character__bottleneck(config, mode='train', scale=5):
    ### env param
    from config.bottleneck import config_env__with_character, config_env__with_character_share
    config.set('envs', [config_env__with_character, config_env__with_character_share] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)



def ray_isac_adaptive_character__bottleneck(config, mode='train', scale=1):
    ### env param
    from config.bottleneck import config_env__with_character, config_env__with_character_share
    config.set('envs', [config_env__with_character, config_env__with_character_share] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)

def ray_RILMthM__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveMultiAgent as Env
    from core.method_isac_recog import IndependentSAC_recog as Method
    config.action_policy_model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    config.action_policy_model_num = 865800
    ### env param
    from config.bottleneck import config_env as config_bottleneck
    # config_bottleneck.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))

    config.set('envs', [
        config_bottleneck,
    ] *scale)

    ### method param
    from config.method import config_recog_multi_agent as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def ray_RILEnvM__bottleneck(config, mode='train', scale=1):
    from utils.env import EnvInteractiveMultiAgentActSvo as Env
    #todo
    from core.method_recog_action_dynamic import RecogV2 as Method
    ### env param
    from config.bottleneck import config_env__actsvo_multiagent as config_bottleneck
    config.action_policy_model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    config.action_policy_model_num = 865800
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__new_action_config(config))

    config.set('envs', [
        config_bottleneck,
    ] *scale)

    ### method param
    from config.method import config_action_svo_multiagent as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def ray_IL__bottleneck(config, mode='train', scale=1):
    from utils.env import EnvInteractiveMultiAgent as Env
    #todo
    from core.method_supervise import IndependentSACsupervise as Method
    config.action_policy_model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    config.action_policy_model_num = 865800
    ### env param
    from config.bottleneck import config_env as config_bottleneck
    # config_bottleneck.set('config_neural_policy', get_sac__bottleneck__new_action_config(config))

    config.set('envs', [
        config_bottleneck,
    ] *scale)

    ### method param
    from config.method import config_supervise_multi as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)


############################################################################
#### intersection ##########################################################
############################################################################



def ray_isac_no_character__intersection(config, mode='train', scale=10):
    ### env param
    from config.intersection import config_env
    config.set('envs', [config_env] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)



def ray_isac_robust_character__intersection(config, mode='train', scale=5):
    ### env param
    from config.intersection import config_env__with_character, config_env__with_character_share
    config.set('envs', [config_env__with_character, config_env__with_character_share] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)



def ray_isac_adaptive_character__intersection(config, mode='train', scale=5):
    ### env param
    from config.intersection import config_env__with_character, config_env__with_character_share
    config.set('envs', [config_env__with_character, config_env__with_character_share] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)


############################################################################
#### merge  ################################################################
############################################################################

def ray_isac_adaptive_character__merge(config, mode='train', scale=1):
    from universe import EnvInteractiveMultiAgent as Env
    from core.method_isac_v0 import IndependentSAC_v0 as Method
    ### env param
    from config.merge import config_env__with_character, config_env__with_character_share
    config.set('envs', [config_env__with_character, config_env__with_character_share] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)


############################################################################
#### multi scenario ########################################################
############################################################################


def ray_isac_no_character__multi_scenario(config, mode='train', scale=2):
    ### env param
    from config import bottleneck, intersection, merge, roundabout
    config.set('envs', [
        bottleneck.config_env, bottleneck.config_env,
        intersection.config_env, intersection.config_env,
        merge.config_env, merge.config_env,
        roundabout.config_env, roundabout.config_env,
    ] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)



def ray_isac_robust_character__multi_scenario(config, mode='train', scale=2):
    ### env param
    from config import bottleneck, intersection, merge, roundabout
    config.set('envs', [
        bottleneck.config_env__with_character, bottleneck.config_env__with_character_share,
        intersection.config_env__with_character, intersection.config_env__with_character_share,
        merge.config_env__with_character, merge.config_env__with_character_share,
        roundabout.config_env__with_character, roundabout.config_env__with_character_share,
    ] *scale)

    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)



def ray_isac_adaptive_character__multi_scenario(config, mode='train', scale=2):
    ### env param
    from config import bottleneck, intersection, merge, roundabout
    config.set('envs', [
        bottleneck.config_env__with_character, bottleneck.config_env__with_character_share,
        intersection.config_env__with_character, intersection.config_env__with_character_share,
        merge.config_env__with_character, merge.config_env__with_character_share,
        roundabout.config_env__with_character, roundabout.config_env__with_character_share,
    ] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)









################################################################################################
##### multi scenario robust, cooperative #######################################################
################################################################################################





from gallery_sa import get_sac__bottleneck__robust_character_config
from gallery_sa import get_sac__intersection__robust_character_config
from gallery_sa import get_sac__merge__robust_character_config
from gallery_sa import get_sac__roundabout__robust_character_config




def ray_isac_robust_character_copo__multi_scenario(config, mode='train', scale=2):
    ### env param
    from utils.agents_master_copo import NeuralVehicleTuneSVO as neural_vehicle_cls
    from utils.agents_master_copo import AgentListMasterTuneSVO as agents_master_cls
    from utils.reward import RewardFunctionGlobalCoordination as reward_func

    from config.bottleneck import config_env as config_bottleneck
    config_bottleneck.set('neural_vehicle_cls', neural_vehicle_cls)
    config_bottleneck.set('agents_master_cls', agents_master_cls)
    config_bottleneck.set('reward_func', reward_func)
    config_bottleneck.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

    from config.intersection import config_env as config_intersection
    config_intersection.set('neural_vehicle_cls', neural_vehicle_cls)
    config_intersection.set('agents_master_cls', agents_master_cls)
    config_intersection.set('reward_func', reward_func)
    config_intersection.set('config_neural_policy', get_sac__intersection__robust_character_config(config))

    from config.merge import config_env as config_merge
    config_merge.set('neural_vehicle_cls', neural_vehicle_cls)
    config_merge.set('agents_master_cls', agents_master_cls)
    config_merge.set('reward_func', reward_func)
    config_merge.set('config_neural_policy', get_sac__merge__robust_character_config(config))

    from config.roundabout import config_env as config_roundabout
    config_roundabout.set('neural_vehicle_cls', neural_vehicle_cls)
    config_roundabout.set('agents_master_cls', agents_master_cls)
    config_roundabout.set('reward_func', reward_func)
    config_roundabout.set('config_neural_policy', get_sac__roundabout__robust_character_config(config))


    config.set('envs', [
        config_bottleneck, config_bottleneck,
        config_intersection, config_intersection,
        config_merge, config_merge,
        config_roundabout, config_roundabout,
    ] *scale)

    ### method param
    from config.method import config_isac__no_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)












############################################################################
#### debug #################################################################
############################################################################




def debug__ray_isac_adaptive_character__intersection(config, mode='train', scale=1):
    ### env param
    from config.intersection import config_env__with_character
    # config_env__with_character.set('num_vehicles_range', rllib.basic.BaseData(min=48, max=48))
    config.set('envs', [config_env__with_character])

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode)






