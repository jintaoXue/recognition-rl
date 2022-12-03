import rllib
import universe
import ray

import time
import psutil
import torch



def run_one_episode(env, method):
    t1 = time.time()
    env.reset()
    t2 = time.time()
    time_select_action = 0.0
    time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()
            # import pdb; pdb.set_trace()

        tt1 = time.time()
        action = env.action_space.sample()
        tt2 = time.time()
        experience, done, info = env.step(action)
        tt3 = time.time()
        time_select_action += (tt2-tt1)
        time_env_step += (tt3-tt2)

        if done:
            break
    
    env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
    env.writer.add_scalar('time_analysis/select_action', time_select_action, env.step_reset)
    env.writer.add_scalar('time_analysis/step', time_env_step, env.step_reset)
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
    
    import gallery_ma_idm_eval as gallery
    import models_ma
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError



    ################################################################################################
    ##### evaluate, idm ############################################################################
    ################################################################################################


    elif version == 'v4-1':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        # scale = 1
        env_master = gallery.evaluate_ray_isac_idm__bottleneck(config, mode, scale)

    elif version == 'v4-2':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        # scale = 1
        env_master = gallery.evaluate_ray_isac_idm__intersection(config, mode, scale)

    elif version == 'v4-3':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        # scale = 1
        env_master = gallery.evaluate_ray_isac_idm__merge(config, mode, scale)

    elif version == 'v4-4':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        # scale = 1
        env_master = gallery.evaluate_ray_isac_idm__roundabout(config, mode, scale)









    else:
        raise NotImplementedError


    try:
        env_master.create_tasks(None, func=run_one_episode)
        ray.get([t.run.remote(n_iters=config.num_episodes) for t in env_master.tasks])

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()





if __name__ == '__main__':
    main()
    
