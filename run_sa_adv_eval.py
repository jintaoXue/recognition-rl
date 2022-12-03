import rllib
import universe
import ray

import time
import copy
import psutil
import torch



def run_one_episode_adv(env, method):
    t1 = time.time()
    env.reset()
    state = env.state[0].to_tensor().unsqueeze(0)
    t2 = time.time()
    time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()

        action, action_adv = method.select_action(state), method.select_action_adv(state)
        action = action.cpu().numpy()
        action_adv = action_adv.cpu().numpy()
        tt1 = time.time()
        experience, done, info = env.step(action, action_adv)
        tt2 = time.time()
        time_env_step += (tt2-tt1)

        state = env.state[0].to_tensor().unsqueeze(0)
        if done:
            break
    
    env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
    env.writer.add_scalar('time_analysis/step', time_env_step, env.step_reset)
    return


def run_one_episode_non_adv(env, method):
    t1 = time.time()
    env.reset()
    state = env.state[0].to_tensor().unsqueeze(0)
    t2 = time.time()
    time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()

        action = method.select_action(state)
        action = action.cpu().numpy()
        tt1 = time.time()
        experience, done, info = env.step(action)
        tt2 = time.time()
        time_env_step += (tt2-tt1)

        state = env.state[0].to_tensor().unsqueeze(0)
        if done:
            break
    
    env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
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
    
    import gallery_sa_adv_eval as gallery
    import models_sa, models_ma
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError
    
    ################################################################################################
    ##### evaluate, training setting bottleneck ####################################################
    ################################################################################################


    elif version == 'v1-1':
        if mode != 'evaluate':
            raise NotImplementedError
        
        scale = 5
        # scale = 1
        adv = True
        models_sa.sac__bottleneck__duplicity().update(config)
        env_master = gallery.evaluate__duplicity_background__bottleneck(config, mode, scale)

    elif version == 'v1-2':
        if mode != 'evaluate':
            raise NotImplementedError
        
        scale = 5
        # scale = 1
        adv = False
        models_sa.sac__bottleneck__duplicity().update(config)
        env_master = gallery.evaluate__adaptive_background__bottleneck(config, mode, scale)




    elif version == 'v1-3':
        if mode != 'evaluate':
            raise NotImplementedError
        
        scale = 5
        # scale = 1
        adv = False
        models_sa.sac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__adaptive_background__bottleneck(config, mode, scale)


    elif version == 'v1-4':
        if mode != 'evaluate':
            raise NotImplementedError
        
        scale = 5
        # scale = 1
        adv = True
        models_sa.sac__bottleneck__adaptive().update(config)
        config_adv = copy.copy(config)
        models_sa.sac__bottleneck__duplicity().update(config_adv)
        config.set('config_adv', config_adv)
        env_master = gallery.evaluate__adaptive_adv_background__bottleneck(config, mode, scale)






    else:
        raise NotImplementedError
    
    try:
        if adv:
            run_one_episode = run_one_episode_adv
        else:
            run_one_episode = run_one_episode_non_adv
        env_master.create_tasks(func=run_one_episode)
        ray.get([t.run.remote(n_iters=200) for t in env_master.tasks])


    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()





if __name__ == '__main__':
    main()
    
