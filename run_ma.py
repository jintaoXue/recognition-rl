import rllib
import universe
import ray

import time
import psutil
import torch



# @rllib.basic.system.cpu_memory_profile
def run_one_episode(env, method):
    t1 = time.time()
    env.reset()
    state = [s.to_tensor().unsqueeze(0) for s in env.state]
    t2 = time.time()
    time_select_action = 0.0
    time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()

        tt1 = time.time()
        action = ray.get(method.select_actions.remote(state)).cpu().numpy()
        tt2 = time.time()
        experience, done, info = env.step(action)
        tt3 = time.time()
        time_select_action += (tt2-tt1)
        time_env_step += (tt3-tt2)

        method.store.remote(experience, index=env.env_index)
        state = [s.to_tensor().unsqueeze(0) for s in env.state]
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
    
    import gallery_ma as gallery
    import models_ma
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError

    ################################################################################################
    ##### bottleneck ###############################################################################
    ################################################################################################

    elif version == 'v1-1':
        config.description += '--flow_bottleneck'
        writer, env_master, method = gallery.ray_isac_no_character__bottleneck(config, mode)
    
    elif version == 'v1-2':
        config.description += '--copo_bottleneck'
        writer, env_master, method = gallery.ray_isac_robust_character__bottleneck(config, mode)

    elif version == 'v1-3':
        config.description += '--adaptive_bottleneck'
        writer, env_master, method = gallery.ray_isac_adaptive_character__bottleneck(config, mode)
    
    elif version == 'v1-4-0':
        scale = 1
        config.description += '--RILMthM'
        writer, env_master, method = gallery.ray_RILMthM__bottleneck(config, mode, scale)
    
    elif version == 'v1-4-0-1':
        scale = 1
        config.description += '--RILMthM_woattn'
        writer, env_master, method = gallery.ray_RILMthM_woattn__bottleneck(config, mode, scale)

    elif version == 'v1-4-1':
        scale = 2
        config.description += '--RILEnvM'
        writer, env_master, method = gallery.ray_RILEnvM__bottleneck(config, mode, scale)
    
    elif version == 'v1-4-2':
        scale = 10
        config.description += '--IL-close-loop'
        writer, env_master, method = gallery.ray_IL__bottleneck(config, mode, scale)
    
    elif version == 'v1-4-2-1':
        scale = 10
        config.description += '--IL-close-loop_woattn'
        writer, env_master, method = gallery.ray_IL_woattn__bottleneck(config, mode, scale)
    
    elif version == 'v1-4-3': 
        scale = 10
        config.description += '--IL-open-loop'
        writer, env_master, method = gallery.ray_IL_open_loop__bottleneck(config, mode, scale)

        env_master.create_tasks(method, func=run_one_episode)

        for i_episode in range(10000):
            total_steps = ray.get([t.run.remote() for t in env_master.tasks])
            buffer_len = ray.get(method.get_buffer_len.remote())
            print('totall step in {} episode: {}, data_size = {}'.format(i_episode, sum(total_steps), buffer_len))
            start_training_step = ray.get(method.get_start_timesteps.remote()) 

            if buffer_len >= start_training_step:
                batch_size = ray.get(method.get_batch_size.remote())
                sample_reuse = ray.get(method.get_sample_reuse.remote())
                n_iters = int(start_training_step / batch_size )*sample_reuse
                print('open loop:update parameter start, buffer_len:{}, update_iters:{}'.format(buffer_len, n_iters))
                ray.get(method.update_parameters_.remote(i_episode, n_iters))
                ray.get(method.close.remote())
                ray.shutdown()
                return

    elif version == 'v1-4-3-1': 
        scale = 10
        config.description += '--IL-open-loop_woattn'
        writer, env_master, method = gallery.ray_IL_open_loop__woattn__bottleneck(config, mode, scale)

        env_master.create_tasks(method, func=run_one_episode)

        for i_episode in range(10000):
            total_steps = ray.get([t.run.remote() for t in env_master.tasks])

            print('totall step in {} episode: {}'.format(i_episode, total_steps))
            buffer_len = ray.get(method.get_buffer_len.remote())
            start_training_step = ray.get(method.get_start_timesteps.remote()) 

            if buffer_len >= start_training_step:
                batch_size = ray.get(method.get_batch_size.remote())
                sample_reuse = ray.get(method.get_sample_reuse.remote())
                n_iters = int(start_training_step / batch_size )*sample_reuse
                print('open loop:update parameter start, buffer_len:{}, update_iters:{}'.format(buffer_len, n_iters))
                ray.get(method.update_parameters_.remote(i_episode, n_iters))
                ray.get(method.close.remote())
                ray.shutdown()
                return
    ################################################################################################
    ##### intersection #############################################################################
    ################################################################################################

    elif version == 'v2-1':
        writer, env_master, method = gallery.ray_isac_no_character__intersection(config, mode)

    elif version == 'v2-2':
        writer, env_master, method = gallery.ray_isac_robust_character__intersection(config, mode)

    elif version == 'v2-3':
        writer, env_master, method = gallery.ray_isac_adaptive_character__intersection(config, mode)
    

    ################################################################################################
    ##### merge ####################################################################################
    ################################################################################################
    
    elif version == 'v3-1':
        writer, env_master, method = gallery.ray_isac_adaptive_character__merge(config, mode)
    
    elif version == 'v3-4-0':
        scale = 9
        config.description += '--RILMthM_merge'
        writer, env_master, method = gallery.ray_RILMthM__merge(config, mode, scale)
    
    elif version == 'v3-4-1':
        scale = 9
        config.description += '--IL_close_loop_merge'
        writer, env_master, method = gallery.ray_IL__merge(config, mode, scale)

    elif version == 'v3-4-2':
        scale = 9
        config.description += '--IL_open_loop__merge'
        writer, env_master, method = gallery.ray_IL_open_loop__merge(config, mode, scale)
    ################################################################################################
    ##### roundabout ###############################################################################
    ################################################################################################




    ################################################################################################
    ##### multi scenario ###########################################################################
    ################################################################################################


    elif version == 'v5-1':
        writer, env_master, method = gallery.ray_isac_no_character__multi_scenario(config, mode)

    elif version == 'v5-2':
        writer, env_master, method = gallery.ray_isac_robust_character__multi_scenario(config, mode)

    elif version == 'v5-3':
        writer, env_master, method = gallery.ray_isac_adaptive_character__multi_scenario(config, mode)
    



    ################################################################################################
    ##### multi scenario robust, cooperative #######################################################
    ################################################################################################


    elif version == 'v6-1':
        writer, env_master, method = gallery.ray_isac_robust_character_copo__multi_scenario(config, mode)







    ################################################################################################
    ##### debug ####################################################################################
    ################################################################################################



    elif version == 'v10':
        scale = 1
        writer, env_master, method = gallery.debug__ray_isac_adaptive_character__intersection(config, mode, scale)

    else:
        raise NotImplementedError


    try:
        env_master.create_tasks(method, func=run_one_episode)

        for i_episode in range(10000):
            total_steps = ray.get([t.run.remote() for t in env_master.tasks])
            print('totall step in {} episode: {}'.format(i_episode, total_steps))
            ray.get(method.update_parameters_.remote(i_episode, n_iters=sum(total_steps)))


    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.get(method.close.remote())
        ray.shutdown()





if __name__ == '__main__':
    main()
    
