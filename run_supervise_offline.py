
import rllib
import universe
import ray
from typing import Tuple
import time
import psutil
import torch
import copy



def init(config, mode, Env, Method) -> Tuple[rllib.basic.Writer, universe.EnvMaster_v0, rllib.template.Method]:
    repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.Writer
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from universe import EnvMaster_v0 as EnvMaster
    # env_master = EnvMaster(config, writer, env_cls=Env)

    # config_method = env_master.config_methods[0]
    # Method = ray.remote(num_cpus=config_method.num_cpus, num_gpus=config_method.num_gpus)(Method).remote
    num_envs = len(config.envs)
    config_env = config.envs[0]
    config.set('dim_state', config_env.perception_cls.dim_state)
    config.set('dim_action', config_env.neural_vehicle_cls.dim_action)
    config_method = config.methods[0]

    cfg_method: rllib.basic.YamlConfig = copy.copy(config_method)
    cfg_method.set('num_workers', num_envs)
    cfg_method.set('path_pack', config.path_pack)
    cfg_method.set('dataset_name', config.dataset_name)
    cfg_method.set('dim_state', config.dim_state)
    cfg_method.set('dim_action', config.dim_action)
    cfg_method.set('evaluate', config.evaluate)
    cfg_method.set('model_dir', config.model_dir)
    cfg_method.set('model_num', config.model_num)
    cfg_method.set('method', config.method)
    cfg_method.set('training_data_path', config.training_data_path)
    method = Method(cfg_method, writer)
    
    # method.reset_writer.remote()

    return writer, method


def ray_supervise_offline__new_adaptive_background__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    #todo
    from core.method_supervise_offline import IndependentSACsupervise as Method

    ### env param
    from config.bottleneck import config_env__neural_background_same_other_svo as config_bottleneck
    
    from gallery_sa import get_sac__new_bottleneck__adaptive_character_config
    config_bottleneck.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))
    config.set('training_data_path', f'./results/data_offline/bottleneck/')
    config.set('envs', [
        config_bottleneck
    ])

    ### method param
    from config.method import config_supervise as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)
def ray_supervise_offline_woattn__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    #todo
    from core.method_supervise_offline import IndependentSACsupervise as Method

    ### env param
    from config.bottleneck import config_env__neural_background_same_other_svo as config_bottleneck
    
    from gallery_sa import get_sac__new_bottleneck__adaptive_character_config
    config_bottleneck.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))
    config.set('training_data_path', f'./results/data_offline/bottleneck/')
    config.set('envs', [
        config_bottleneck
    ])

    ### method param
    from config.method import config_woattn as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def main():
    config = rllib.basic.YamlConfig()
    from config.args import generate_args
    args = generate_args()
    config.update(args)

    # ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)

    mode = 'train'
    if config.evaluate == True:
        mode = 'evaluate'
        config.seed += 1
    rllib.basic.setup_seed(config.seed)
    
    import gallery_sa as gallery
    import models_sa, models_ma
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError
    
    elif version == 'v6-5-6':
        scale = 10
        config.description += '--supervise-offline-hrz10-act1'
        writer, method = ray_supervise_offline__new_adaptive_background__bottleneck(config, mode, scale)
    
    elif version == 'v6-5-7':
        scale = 10
        config.description += '--supervise-offline-woattn'
        writer, method = ray_supervise_offline_woattn__bottleneck(config, mode, scale)

    # try:
    #     env_master.create_tasks(method, func=run_one_episode)

    #     for i_episode in range(10000):
    #         total_steps = ray.get([t.run.remote() for t in env_master.tasks])
    #         print('update episode i_episode: ', i_episode)
    #         ray.get(method.update_parameters_.remote(i_episode, n_iters=sum(total_steps)))

    for i_episode in range(10000):
        print('update episode i_episode: ', i_episode)      
        method.update_parameters_(index=i_episode,n_iters=1000)

    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    # finally:
    #     ray.get(method.close.remote())
    #     ray.shutdown()


if __name__ == '__main__':
    main()