import rllib
import universe
import ray

import os
import copy

import torch




def init(config, mode, Env, Method) -> universe.EnvMaster_v1:
    # repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    # config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)

    from core import method_evaluate
    if config.method == 'sac':
        Method = method_evaluate.EvaluateSAC
    elif config.method == 'ppo':
        Method = method_evaluate.EvaluatePPO
    elif config.method == 'IndependentSAC_v0':
        Method = method_evaluate.EvaluateIndependentSAC
    elif config.method == 'IndependentSAC_recog':
        Method = method_evaluate.EvaluateSACRecog
    elif config.method == 'IndependentSAC_supervise' or config.method == 'IndependentSACsupervise':
        Method = method_evaluate.EvaluateSACsupervise
    else:
        raise NotImplementedError
    
    from universe import EnvMaster_v1 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env, method_cls=Method)
    return env_master

def init_fix_svo(config, mode, Env,Method, ego_svo, other_svo):

    repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)

    from core import method_evaluate
    if config.method == 'sac':
        Method = method_evaluate.EvaluateSAC
    elif config.method == 'ppo':
        Method = method_evaluate.EvaluatePPO
    elif config.method == 'IndependentSAC_v0':
        Method = method_evaluate.EvaluateIndependentSAC
    elif config.method == 'IndependentSAC_recog':
        Method = method_evaluate.EvaluateSACRecog
    elif config.method == 'IndependentSAC_supervise':
        Method = method_evaluate.EvaluateSupervise
    else:
        raise NotImplementedError
    
    from universe import EnvMaster_v2 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env, method_cls=Method, ego_svo= ego_svo, other_svo=other_svo)
    return env_master


from gallery_sa import get_sac__new_bottleneck__adaptive_character_config
from gallery_sa import get_sac__bottleneck__adaptive_character_config
from gallery_sa import get_sac__intersection__adaptive_character_config
from gallery_sa import get_sac__merge__adaptive_character_config
from gallery_sa import get_sac__roundabout__adaptive_character_config

from gallery_sa import get_sac__bottleneck__robust_character_config
from gallery_sa import get_sac__intersection__robust_character_config
from gallery_sa import get_sac__merge__robust_character_config
from gallery_sa import get_sac__roundabout__robust_character_config

from gallery_sa import get_sac__bottleneck__no_character_config
from gallery_sa import get_sac__intersection__no_character_config
from gallery_sa import get_sac__merge__no_character_config
from gallery_sa import get_sac__roundabout__no_character_config

def evaluate_ray_sac__bottleneck__idm_background(config, mode='evalute', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.model_vectornet import SAC as Method

    ### env param
    from config.bottleneck_evaluate import config_env__idm_background
    config_env__idm_background.set('num_steps', 200)
    # config_env__neural_background.scenario_name += '_adaptive_character'
    config.set('envs', [config_env__idm_background] *scale)
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)
def evaluate_ray_sac__bottleneck__adaptive_background(config, mode='train', scale=10):
    from universe import EnvInteractiveSingleAgent as Env
    from core.model_vectornet import SAC as Method

    ### env param
    from config.bottleneck_evaluate import config_env__neural_background
    config_env__neural_background.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))
    config_env__neural_background.scenario_name += '_adaptive_character'
    config.set('envs', [config_env__neural_background] *scale)
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def evaluate_ray_ppo__bottleneck__adaptive_background(config, mode='evalute', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.model_vectornet import PPO as Method

    ### env param
    from config.bottleneck_evaluate import config_env__neural_background
    config_env__neural_background.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))
    # config_env__neural_background.scenario_name += '_adaptive_character'
    config.set('envs', [config_env__neural_background] *scale)
    
    ### method param
    from config.method import config_ppo as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)


def evaluate_ray_sac__bottleneck__multi_background(config, mode='train', scale=5):
    from universe import EnvInteractiveSingleAgent as Env
    from core.model_vectornet import SAC as Method

    ### env param
    from config.bottleneck_evaluate import config_env__neural_background
    from config.bottleneck_evaluate import config_env__idm_background
    config_env__neural_background.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))
    config_env__neural_background.scenario_name += '_adaptive_character'
    config.set('envs', [config_env__neural_background] *scale + [config_env__idm_background] *scale)
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def evaluate_ray_ppo__bottleneck__idm_background(config, mode='evalute', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.model_vectornet import PPO as Method

    ### env param
    from config.bottleneck_evaluate import config_env__idm_background
    config_env__idm_background.set('num_steps', 200)
    # config_env__neural_background.scenario_name += '_adaptive_character'
    config.set('envs', [config_env__idm_background] *scale)
    
    ### method param
    from config.method import config_ppo as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)


################################################################################################
##### evaluate, full stack #####################################################################
################################################################################################



def evaluate__full_stack_background__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.model_vectornet import SAC as Method

    ### env param
    from config.bottleneck_evaluate import config_env__idm_background
    from config.bottleneck_evaluate import config_env__neural_background

    ### idm
    config_env__idm = config_env__idm_background

    ### no character
    config_env__no_svo = copy.deepcopy(config_env__neural_background)
    config_env__no_svo.set('config_neural_policy', get_sac__bottleneck__no_character_config(config))

    ### robust copo
    config_env__copo = copy.deepcopy(config_env__neural_background)
    config_env__copo.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
    config_env__copo.set('agents_master_cls', agents_master_cls)

    ### explicit adv
    config_env__copo_adv = copy.deepcopy(config_env__neural_background)
    config_env__copo_adv.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPOAdv as agents_master_cls
    config_env__copo_adv.set('agents_master_cls', agents_master_cls)

    ### adaptive
    config_env__adaptive = copy.deepcopy(config_env__neural_background)
    config_env__adaptive.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))

    ### duplicity
    config_env__duplicity = copy.deepcopy(config_env__neural_background)
    config_env__duplicity.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))

    from utils.scenarios_bottleneck import ScenarioBottleneckEvaluate as scenario_cls
    config_env__duplicity.set('scenario_cls', scenario_cls)

    config.set('envs', [
        config_env__idm,
        config_env__no_svo,
        config_env__copo,
        config_env__copo_adv,
        config_env__adaptive,
        config_env__duplicity,
    ])
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)









def evaluate__full_stack_background__intersection(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.model_vectornet import SAC as Method

    ### env param
    from config.intersection_evaluate import config_env__idm_background
    from config.intersection_evaluate import config_env__neural_background
    config_env__idm_background.set('scenario_name', 'intersection_v2')
    config_env__neural_background.set('scenario_name', 'intersection_v2')

    ### idm
    config_env__idm = config_env__idm_background

    ### no character
    config_env__no_svo = copy.deepcopy(config_env__neural_background)
    config_env__no_svo.set('config_neural_policy', get_sac__intersection__no_character_config(config))

    ### robust copo
    config_env__copo = copy.deepcopy(config_env__neural_background)
    config_env__copo.set('config_neural_policy', get_sac__intersection__robust_character_config(config))

    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
    config_env__copo.set('agents_master_cls', agents_master_cls)

    ### explicit adv
    config_env__copo_adv = copy.deepcopy(config_env__neural_background)
    config_env__copo_adv.set('config_neural_policy', get_sac__intersection__robust_character_config(config))

    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPOAdv as agents_master_cls
    config_env__copo_adv.set('agents_master_cls', agents_master_cls)

    ### adaptive
    config_env__adaptive = copy.deepcopy(config_env__neural_background)
    config_env__adaptive.set('config_neural_policy', get_sac__intersection__adaptive_character_config(config))

    ### duplicity
    config_env__duplicity = copy.deepcopy(config_env__neural_background)
    config_env__duplicity.set('config_neural_policy', get_sac__intersection__adaptive_character_config(config))

    from utils.scenarios_intersection import ScenarioIntersectionEvaluate as scenario_cls
    config_env__duplicity.set('scenario_cls', scenario_cls)

    config.set('envs', [
        config_env__idm,
        config_env__no_svo,
        config_env__copo,
        config_env__copo_adv,
        config_env__adaptive,
        config_env__duplicity,
    ])
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)







def evaluate__full_stack_background__merge(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.model_vectornet import SAC as Method

    ### env param
    from config.merge_evaluate import config_env__idm_background
    from config.merge_evaluate import config_env__neural_background

    ### idm
    config_env__idm = config_env__idm_background

    ### no character
    config_env__no_svo = copy.deepcopy(config_env__neural_background)
    config_env__no_svo.set('config_neural_policy', get_sac__merge__no_character_config(config))

    ### robust copo
    config_env__copo = copy.deepcopy(config_env__neural_background)
    config_env__copo.set('config_neural_policy', get_sac__merge__robust_character_config(config))

    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
    config_env__copo.set('agents_master_cls', agents_master_cls)

    ### explicit adv
    config_env__copo_adv = copy.deepcopy(config_env__neural_background)
    config_env__copo_adv.set('config_neural_policy', get_sac__merge__robust_character_config(config))

    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPOAdv as agents_master_cls
    config_env__copo_adv.set('agents_master_cls', agents_master_cls)

    ### adaptive
    config_env__adaptive = copy.deepcopy(config_env__neural_background)
    config_env__adaptive.set('config_neural_policy', get_sac__merge__adaptive_character_config(config))

    ### duplicity
    config_env__duplicity = copy.deepcopy(config_env__neural_background)
    config_env__duplicity.set('config_neural_policy', get_sac__merge__adaptive_character_config(config))

    from utils.scenarios_merge import ScenarioMergeEvaluate as scenario_cls
    config_env__duplicity.set('scenario_cls', scenario_cls)

    config.set('envs', [
        config_env__idm,
        config_env__no_svo,
        config_env__copo,
        config_env__copo_adv,
        config_env__adaptive,
        config_env__duplicity,
    ])
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)








def evaluate__full_stack_background__roundabout(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.model_vectornet import SAC as Method

    ### env param
    from config.roundabout_evaluate import config_env__idm_background
    from config.roundabout_evaluate import config_env__neural_background

    ### idm
    config_env__idm = config_env__idm_background

    ### no character
    config_env__no_svo = copy.deepcopy(config_env__neural_background)
    config_env__no_svo.set('config_neural_policy', get_sac__roundabout__no_character_config(config))

    ### robust copo
    config_env__copo = copy.deepcopy(config_env__neural_background)
    config_env__copo.set('config_neural_policy', get_sac__roundabout__robust_character_config(config))

    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
    config_env__copo.set('agents_master_cls', agents_master_cls)

    ### explicit adv
    config_env__copo_adv = copy.deepcopy(config_env__neural_background)
    config_env__copo_adv.set('config_neural_policy', get_sac__roundabout__robust_character_config(config))

    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPOAdv as agents_master_cls
    config_env__copo_adv.set('agents_master_cls', agents_master_cls)

    ### adaptive
    config_env__adaptive = copy.deepcopy(config_env__neural_background)
    config_env__adaptive.set('config_neural_policy', get_sac__roundabout__adaptive_character_config(config))

    ### duplicity
    config_env__duplicity = copy.deepcopy(config_env__neural_background)
    config_env__duplicity.set('config_neural_policy', get_sac__roundabout__adaptive_character_config(config))

    from utils.scenarios_roundabout import ScenarioRoundaboutEvaluate as scenario_cls
    config_env__duplicity.set('scenario_cls', scenario_cls)

    config.set('envs', [
        config_env__idm,
        config_env__no_svo,
        config_env__copo,
        config_env__copo_adv,
        config_env__adaptive,
        config_env__duplicity,
    ])
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)



def evaluate__isac_roubust__four_background__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.method_isac_v0 import IndependentSAC_v0 as Method

    ### env param
    from config.bottleneck_evaluate import config_env__idm_background
    from config.bottleneck_evaluate import config_env__neural_background

    ### idm
    config_env__idm = config_env__idm_background

    ### no character
    config_env__no_svo = copy.deepcopy(config_env__neural_background)
    config_env__no_svo.set('config_neural_policy', get_sac__bottleneck__no_character_config(config))

    ### robust copo
    config_env__copo = copy.deepcopy(config_env__neural_background)
    config_env__copo.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

    from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
    config_env__copo.set('agents_master_cls', agents_master_cls)

    ### adaptive
    config_env__adaptive = copy.deepcopy(config_env__neural_background)
    config_env__adaptive.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))



    config.set('envs', [
        config_env__idm,
        config_env__no_svo,
        config_env__copo,
        config_env__adaptive,
    ])
    
    ### method param
    from config.method import config_isac__robust_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def evaluate__isac_adaptive__four_background__bottleneck(config, mode='train', scale=1):
        from universe import EnvInteractiveSingleAgent as Env
        from core.method_isac_v0 import IndependentSAC_v0 as Method

        ### env param
        from config.bottleneck_evaluate import config_env__idm_background
        from config.bottleneck_evaluate import config_env__neural_background

        ### idm
        config_env__idm = config_env__idm_background

        ### no character
        config_env__no_svo = copy.deepcopy(config_env__neural_background)
        config_env__no_svo.set('config_neural_policy', get_sac__bottleneck__no_character_config(config))

        ### robust copo
        config_env__copo = copy.deepcopy(config_env__neural_background)
        config_env__copo.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

        from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
        config_env__copo.set('agents_master_cls', agents_master_cls)

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background)
        config_env__adaptive.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))



        config.set('envs', [
            config_env__idm,
            config_env__no_svo,
            config_env__copo,
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_isac__adaptive_character as config_method
        config.set('methods', [config_method])

        return init(config, mode, Env, Method)

def evaluate__isac_adaptive__adaptive_background__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.method_isac_v0 import IndependentSAC_v0 as Method

    ### env param
    from config.bottleneck_evaluate import config_env__idm_background
    from config.bottleneck_evaluate import config_env__neural_background

    ### adaptive
    config_env__adaptive = copy.deepcopy(config_env__neural_background)
    config_env__adaptive.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))

    config.set('envs', [
        config_env__adaptive,
    ])
    
    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def evaluate__isac__adaptive_background_downsample_bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.method_isac_v0 import IndependentSAC_v0 as Method

    ### env param

    from config.bottleneck_evaluate import config_env__neural_background_sampling

    ### adaptive
    config_env__adaptive = copy.deepcopy(config_env__neural_background_sampling)
    config_env__adaptive.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))

    config.set('envs', [
        config_env__adaptive,
    ])
    
    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def evaluate__sac__four_background__bottleneck(config, mode='train', scale=1):
        from universe import EnvInteractiveSingleAgent as Env
        from core.model_vectornet import SAC as Method

        ### env param
        from config.bottleneck_evaluate import config_env__idm_background
        from config.bottleneck_evaluate import config_env__neural_background

        ### idm
        config_env__idm = config_env__idm_background

        ### no character
        config_env__no_svo = copy.deepcopy(config_env__neural_background)
        config_env__no_svo.set('config_neural_policy', get_sac__bottleneck__no_character_config(config))

        ### robust copo
        config_env__copo = copy.deepcopy(config_env__neural_background)
        config_env__copo.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

        from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
        config_env__copo.set('agents_master_cls', agents_master_cls)

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background)
        config_env__adaptive.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))



        config.set('envs', [
            config_env__idm,
            config_env__no_svo,
            config_env__copo,
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_sac as config_method
        config.set('methods', [config_method])

        return init(config, mode, Env, Method)

def evaluate__isac_recog__idm_background__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.method_isac_recog import IndependentSAC_recog as Method

    ### env param
    from config.bottleneck_evaluate import config_env__idm_background

    ### idm
    config_env__idm = config_env__idm_background

    config.set('envs', [
        config_env__idm
    ])
    
    ### method param
    from config.method import config_isac_recog as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def evaluate__isac_recog__adaptive_background__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    from core.method_isac_recog import IndependentSAC_recog as Method

    ### env param
    from config.bottleneck_evaluate import config_env__neural_background
    config_env__adaptive = copy.deepcopy(config_env__neural_background)
    config_env__adaptive.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))

    config.set('envs', [
        config_env__adaptive
    ])
    
    ### method param
    from config.method import config_isac_recog as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def evaluate__isac_recog__one_background__bottleneck(config, mode='train', scale=1):
        from universe import EnvInteractiveSingleAgent as Env
        from core.method_isac_recog import IndependentSAC_recog as Method

        from config.bottleneck_evaluate import config_env__neural_background_same_other_svo

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background_same_other_svo)
        config_env__adaptive.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))

        config.set('envs', [
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_isac_recog as config_method
        config.set('methods', [config_method])

        return init(config, mode, Env, Method)

def evaluate__recog__one_background_downsample__bottleneck(config, mode='train', scale=1):
        from universe import EnvInteractiveSingleAgent as Env
        from core.method_isac_recog import IndependentSAC_recog as Method

        from config.bottleneck_evaluate import config_env__neural_background_sampling

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background_sampling)
        config_env__adaptive.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))

        config.set('envs', [
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_isac_recog as config_method
        config.set('methods', [config_method])

        return init(config, mode, Env, Method)

def evaluate__isac_recog__four_background__bottleneck(config, mode='train', scale=1):
        from universe import EnvInteractiveSingleAgent as Env
        from core.method_isac_recog import IndependentSAC_recog as Method

        ### env param
        from config.bottleneck_evaluate import config_env__idm_background
        from config.bottleneck_evaluate import config_env__neural_background_sampling

        ### idm
        config_env__idm = config_env__idm_background

        ### no character
        config_env__no_svo = copy.deepcopy(config_env__neural_background_sampling)
        config_env__no_svo.set('config_neural_policy', get_sac__bottleneck__no_character_config(config))

        ### robust copo
        config_env__copo = copy.deepcopy(config_env__neural_background_sampling)
        config_env__copo.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

        from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
        config_env__copo.set('agents_master_cls', agents_master_cls)

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background_sampling)
        config_env__adaptive.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))



        config.set('envs', [
            config_env__idm,
            config_env__no_svo,
            config_env__copo,
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_isac_recog as config_method
        config.set('methods', [config_method])

        return init(config, mode, Env, Method)
        
def evaluate__recog_random_svo_one_background__bottleneck(config, mode='train', scale=1):
        from universe import EnvInteractiveSingleAgent as Env
        from core.method_isac_recog import IndependentSAC_recog as Method

        from config.bottleneck_evaluate import config_env__neural_background

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background)
        config_env__adaptive.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))

        config.set('envs', [
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_isac_recog as config_method
        config.set('methods', [config_method])

        return init(config, mode, Env, Method)

def evaluate__fix_svo__four_background__bottleneck(config, mode='train', scale=1):
        from universe import EnvInteractiveSingleAgent as Env
        from core.method_isac_v0 import IndependentSAC_v0 as Method

        ### env param
        from config.bottleneck_evaluate import config_env__idm_background_fix
        from config.bottleneck_evaluate import config_env__neural_background_fix

        ### idm
        config_env__idm = config_env__idm_background_fix

        ### no character
        config_env__no_svo = copy.deepcopy(config_env__neural_background_fix)
        config_env__no_svo.set('config_neural_policy', get_sac__bottleneck__no_character_config(config))

        ### robust copo
        config_env__copo = copy.deepcopy(config_env__neural_background_fix)
        config_env__copo.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

        from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
        config_env__copo.set('agents_master_cls', agents_master_cls)

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background_fix)
        config_env__adaptive.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))



        config.set('envs', [
            config_env__idm,
            config_env__no_svo,
            config_env__copo,
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_isac__adaptive_character as config_method
        config.set('methods', [config_method])

        return init(config, mode, Env, Method)

def evaluate__fix_svo__two_background__bottleneck(config, ego_svo, other_svo, mode='train',scale=1):
        from universe import EnvInteractiveSingleAgentFixSvo as Env

        from core.method_isac_v0 import IndependentSAC_v0 as Method

        ### env param
        from config.bottleneck_evaluate import config_env__idm_background_fix
        from config.bottleneck_evaluate import config_env__neural_background_fix

        ### robust copo
        config_env__copo = copy.deepcopy(config_env__neural_background_fix)
        config_env__copo.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

        from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
        config_env__copo.set('agents_master_cls', agents_master_cls)

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background_fix)
        config_env__adaptive.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))


        config.set('envs', [
            config_env__copo,
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_isac__adaptive_character as config_method
        config.set('methods', [config_method])

        return init_fix_svo(config, mode, Env,Method, ego_svo, other_svo)
def evaluate__fix_svo__new_one_background__bottleneck(config, ego_svo, other_svo, mode='train',scale=1):
        from universe import EnvInteractiveSingleAgentFixSvo as Env

        from core.method_isac_v0 import IndependentSAC_v0 as Method

        ### env param
        from config.bottleneck_evaluate import config_env__idm_background_fix
        from config.bottleneck_evaluate import config_env__neural_background_fix

        # ### robust copo
        # config_env__copo = copy.deepcopy(config_env__neural_background_fix)
        # config_env__copo.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

        # from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
        # config_env__copo.set('agents_master_cls', agents_master_cls)

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background_fix)
        config_env__adaptive.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))


        config.set('envs', [
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_isac__adaptive_character as config_method
        config.set('methods', [config_method])

        return init_fix_svo(config, mode, Env,Method, ego_svo, other_svo)


def evaluate__supervise__four_background__bottleneck(config, mode='train', scale=1):
        from universe import EnvInteractiveSingleAgent as Env
        from core.method_supervise import IndependentSACsupervise as Method

        ### env param
        from config.bottleneck_evaluate import config_env__idm_background
        from config.bottleneck_evaluate import config_env__neural_background

        ### idm
        config_env__idm = config_env__idm_background

        ### no character
        config_env__no_svo = copy.deepcopy(config_env__neural_background)
        config_env__no_svo.set('config_neural_policy', get_sac__bottleneck__no_character_config(config))

        ### robust copo
        config_env__copo = copy.deepcopy(config_env__neural_background)
        config_env__copo.set('config_neural_policy', get_sac__bottleneck__robust_character_config(config))

        from utils.agents_master import AgentListMasterNeuralBackgroundManualTuneSVOCoPO as agents_master_cls
        config_env__copo.set('agents_master_cls', agents_master_cls)

        ### adaptive
        config_env__adaptive = copy.deepcopy(config_env__neural_background)
        config_env__adaptive.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))



        config.set('envs', [
            config_env__idm,
            config_env__no_svo,
            config_env__copo,
            config_env__adaptive,
        ])
        
        ### method param
        from config.method import config_supervise as config_method
        config.set('methods', [config_method])

        return init(config, mode, Env, Method)