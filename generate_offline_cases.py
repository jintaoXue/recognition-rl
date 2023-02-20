import rllib
import ray

import os
import time
import tqdm
import pickle
import psutil
import torch
import copy


def run_one_episode(env, method):
    env.reset()

    dir_path = f'./results/scenario_offline/{env.config.scenario_name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, str(env.step_reset) + '.txt')
    print('file path: ', file_path, env.scenario.scenario_randomization.num_vehicles)
    with open(file_path, 'wb') as f:
        pickle.dump(env.scenario.scenario_randomization, f)
    return



if __name__ == "__main__":
    config = rllib.basic.YamlConfig()
    from config.args import generate_args
    args = generate_args()
    config.update(args)

    ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)

    mode = 'train'
    if config.evaluate == True:
        mode = 'evaluate'
        config.seed += 0
    rllib.basic.setup_seed(config.seed)

    version = config.version
    
    from universe import EnvInteractiveMultiAgent as Env
    # from universe import EnvInteractiveSingleAgent as Env
    from config import bottleneck, intersection, merge, roundabout
    # config.set('envs', [
    #     bottleneck.config_env__with_character,

    # ])
    # config.set('envs', [
    #     bottleneck.config_env__with_character,
    #     intersection.config_env__with_character,
    #     merge.config_env__with_character,
    #     roundabout.config_env__with_character,
    # ])

    ###multi agent
    if version == 'v1-4':
        config.set('envs', [
            bottleneck.config_env,
        ])
    if version == 'v1-5':
        config.set('envs', [
            merge.config_env,
        ])
    elif version == 'v6-6-2':
        config.set('envs', [
            bottleneck.config_env__with_character_fix_other_svo,
        ])
    ###mix background
    elif version == 'v6-6-2-1':
        import gallery_sa
        config_env__adaptive = copy.deepcopy(bottleneck.config_env__multiact__mixbkgrd)
        config_env__adaptive.set('config_neural_policy_ours', gallery_sa.get_sac__bottleneck__new_action_config(config))
        config_env__adaptive.set('config_neural_policy_robust', gallery_sa.get_sac__bottleneck__robust_character_config(config))
        config_env__adaptive.set('config_neural_policy_flow', gallery_sa.get_sac__bottleneck__no_character_config(config))

        config.set('envs', [
            config_env__adaptive,
        ])
    
    for env in config.envs:
        env.set('num_vehicles_range', rllib.basic.BaseData(min=20, max=20))
    config.set('methods', [])


    model_name = 'PseudoMethod' + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)

    from universe import EnvMaster_v0 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env)



    try:
        env_master.create_tasks(method=None, func=run_one_episode)

        ray.get([t.run.remote(n_iters=2000) for t in env_master.tasks])

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


