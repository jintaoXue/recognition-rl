import rllib
import universe
import ray

import time
import psutil
import torch



def run_one_episode(env):
    env.reset()
    while True:
        if env.config.render:
            env.render()

        action = env.action_space.sample()
        # action[:] = -1
        experience, done, info = env.step(action)
        if done:
            break    
    return




def main():
    config = rllib.basic.YamlConfig()
    from config.args import generate_args
    args = generate_args()
    config.update(args)

    ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)

    mode = 'train'
    if config.evaluate == True:
        mode = 'evaluate'
        config.seed += 1
    rllib.basic.setup_seed(config.seed)
    
    import gallery_env as gallery
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError

    ################################################################################################
    ##### bottleneck ###############################################################################
    ################################################################################################

    elif version == 'v1-1':
        scale = 1
        env_master = gallery.ray_sac__bottleneck__idm_background(config, mode, scale)




    else:
        raise NotImplementedError
    
    try:
        env_master.create_envs()
        env = env_master.envs[0]

        for i in range(10000):
            run_one_episode(env)


    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        pass





if __name__ == '__main__':
    main()
    
