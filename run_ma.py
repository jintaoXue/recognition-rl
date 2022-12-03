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
        writer, env_master, method = gallery.ray_isac_no_character__bottleneck(config, mode)
    
    elif version == 'v1-2':
        writer, env_master, method = gallery.ray_isac_robust_character__bottleneck(config, mode)

    elif version == 'v1-3':
        writer, env_master, method = gallery.ray_isac_adaptive_character__bottleneck(config, mode)
    

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
            print('update episode i_episode: ', i_episode)
            ray.get(method.update_parameters_.remote(i_episode, n_iters=sum(total_steps)))


    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.get(method.close.remote())
        ray.shutdown()





if __name__ == '__main__':
    main()
    
