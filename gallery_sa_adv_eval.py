import rllib
import universe
import ray

import os
import copy
import torch





def init(config, mode, Env, Method):
    repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    config.set('github_repos', repos)

    model_name = 'SAC' + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
        
    from universe import EnvMaster_v1 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env, method_cls=Method)
    return env_master




from gallery_sa import get_sac__bottleneck__adaptive_character_config
from gallery_sa import get_sac__intersection__adaptive_character_config
from gallery_sa import get_sac__merge__adaptive_character_config
from gallery_sa import get_sac__roundabout__adaptive_character_config




################################################################################################
##### evaluate, training setting bottleneck ####################################################
################################################################################################



def evaluate__duplicity_background__bottleneck(config, mode='train', scale=5):
    from core.env_sa_adv import EnvSingleAgentAdv as Env
    from core.method_evaluate import EvaluateSACAdv as Method

    ### env param
    from config.bottleneck_evaluate import config_env__neural_background
    config_env__neural_background.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))
    config.set('envs', [config_env__neural_background] *scale)
    
    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)





def evaluate__adaptive_background__bottleneck(config, mode='train', scale=5):
    from universe import EnvInteractiveSingleAgent as Env
    from core.method_evaluate import EvaluateSAC as Method

    ### env param
    from config.bottleneck_evaluate import config_env__neural_background
    config_env__neural_background.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))
    config.set('envs', [config_env__neural_background] *scale)

    ### method param
    from config.method import config_sac as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)




def evaluate__adaptive_adv_background__bottleneck(config, mode='train', scale=5):
    from core.env_sa_adv import EnvSingleAgentAdv as Env
    from core.method_evaluate import EvaluateSACAdvDecouple as Method

    ### env param
    from config.bottleneck_evaluate import config_env__neural_background
    config_env__neural_background.set('config_neural_policy', get_sac__bottleneck__adaptive_character_config(config))
    config.set('envs', [config_env__neural_background] *scale)
    
    ### method param
    from config.method import config_sac as config_method
    config_method.set('config_adv', config.config_adv)
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)



