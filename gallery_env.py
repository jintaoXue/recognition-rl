import rllib
import universe
import ray

import os
import copy
from typing import Tuple

import torch

import models_ma


def init(config, mode, Env) -> universe.EnvMaster_v0:
    repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    config.set('github_repos', repos)

    model_name = 'PseudoMethod' + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from universe import EnvMaster_v0 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env)
    return env_master








def ray_sac__bottleneck__idm_background(config, mode='train', scale=10):
    from universe import EnvInteractiveSingleAgent as Env

    ### env param
    from config.bottleneck import config_env
    config.set('envs', [config_env] *scale)

    config.set('methods', [])

    return init(config, mode, Env)



