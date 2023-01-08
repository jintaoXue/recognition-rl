import rllib
import universe
import ray

import time
import psutil
import torch



def run_one_episode(env, method):
    t1 = time.time()
    env.reset()
    state = env.state[0].to_tensor().unsqueeze(0)
    t2 = time.time()
    time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()

        action = ray.get(method.select_action.remote(state)).cpu().numpy()
        tt1 = time.time()
        experience, done, info = env.step(action)

        tt2 = time.time()
        time_env_step += (tt2-tt1)

        method.store.remote(experience, index=env.env_index)
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
    
    import gallery_sa as gallery
    import models_sa, models_ma
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError

    ################################################################################################
    ##### bottleneck ###############################################################################
    ################################################################################################

    elif version == 'v1-1':
        scale = 1
        writer, env_master, method = gallery.ray_sac__bottleneck__idm_background(config, mode, scale)

    elif version == 'v1-2':
        scale = 16
        writer, env_master, method = gallery.ray_sac__bottleneck__idm_background_our_attn(config, mode, scale)

    elif version == 'v1-4':
        scale = 10
        writer, env_master, method = gallery.ray_sac__bottleneck__adaptive_background(config, mode, scale)
    #xue
    elif version == 'v1-5':     
        scale = 7
        writer, env_master, method = gallery.ray_ppo__bottleneck__adaptive_background(config, mode, scale)

    elif version == 'v1-6':
        scale = 1
        writer, env_master, method = gallery.ray_ppo__bottleneck__adaptive_background_our_attn(config, mode, scale)

    elif version == 'v1-7':
        scale = 10
        writer, env_master, method = gallery.ray_ppo__bottleneck__idm_background(config, mode, scale)

    ################################################################################################
    ##### merge ####################################################################################
    ################################################################################################

    elif version == 'v3-4':
        scale = 10
        scale = 1
        writer, env_master, method = gallery.ray_sac__merge__adaptive_background(config, mode, scale)
    #xue
    elif version == 'v3-5':
        scale = 1
        writer, env_master, method = gallery.ray_ppo__merge__adaptive_background(config, mode, scale)
    #xue
    elif version == 'v3-6':
        scale = 2
        writer, env_master, method = gallery.ray_ppo__merge__idm_background(config, mode, scale)
    ################################################################################################
    ##### roundabout ###############################################################################
    ################################################################################################

    elif version == 'v4-4':
        scale = 10
        scale = 1
        writer, env_master, method = gallery.ray_sac__roundabout__adaptive_background(config, mode, scale)



    ################################################################################################
    ##### multi scenario ###########################################################################
    ################################################################################################


    elif version == 'v5-1':
        writer, env_master, method = gallery.ray_sac__idm_background__multi_scenario(config, mode)

    elif version == 'v5-2':
        writer, env_master, method = gallery.ray_sac__no_character__multi_scenario(config, mode)

    elif version == 'v5-3':
        writer, env_master, method = gallery.ray_sac__robust_character_copo__multi_scenario(config, mode)

    elif version == 'v5-4':
        writer, env_master, method = gallery.ray_sac__robust_character_copo_adv__multi_scenario(config, mode)

    elif version == 'v5-5':
        writer, env_master, method = gallery.ray_sac__adaptive_character__multi_scenario(config, mode)


    ################################################################################################
    ##### character recognition#####################################################################
    ################################################################################################

    elif version == 'v6-1':
        scale = 10
        writer, env_master, method = gallery.ray_isac_recog__idm_background__bottleneck(config, mode, scale)

    elif version == 'v6-2':
        scale = 10
        config.description += '--isac_recog__no_character'
        writer, env_master, method = gallery.ray_isac_recog__no_character_background__bottleneck(config, mode, scale)

    elif version == 'v6-3':
        scale = 10
        config.description += '--isac_recog__robust'
        writer, env_master, method = gallery.ray_isac_recog__robust_background__bottleneck(config, mode, scale)

    elif version == 'v6-4-0':
        scale = 10
        config.description += '--isac_recog__adaptive'
        writer, env_master, method = gallery.ray_isac_recog__adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-4-1':
        scale = 1
        config.description += '--isac_recog__new_adaptive'
        writer, env_master, method = gallery.ray_isac_recog__new_adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-4-2':
        scale = 10
        config.description += '--isac_recog__downsample_new_adaptive'
        writer, env_master, method = gallery.ray_isac_recog__downsample_adaptive_background__bottleneck(config, mode, scale)
    
    elif version == 'v6-4-3':
        scale = 12
        config.description += '--isac_recog_woattn__new_adaptive'
        writer, env_master, method = gallery.ray_isac_recog_woattn__adaptive_background__bottleneck(config, mode, scale)
    
    elif version == 'v6-5-0':
        scale = 10
        config.description += '--supervise-hrz10-act10'
        writer, env_master, method = gallery.ray_supervise__adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-5-1':
        scale = 1
        config.description += '--supervise-hrz10-act1'
        writer, env_master, method = gallery.ray_supervise__adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-5-2':
        scale = 10
        config.description += '--supervise-sampling-trj-hrz10-act10'
        writer, env_master, method = gallery.ray_supervise_sample_trj__adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-5-3':
        scale = 1
        config.description += '--supervise-sampling-hrz10-act10'
        writer, env_master, method = gallery.ray_supervise_sample__adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-5-4':
        scale = 1
        config.description += '--supervise-sampling-hrz10-act10'
        writer, env_master, method = gallery.ray_supervise__new_adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-5-5':
        scale = 1
        config.description += '--supervise-sampling2-hrz10-act10'
        writer, env_master, method = gallery.ray_supervise_sampling__new_adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-5-6':
        scale = 10
        config.description += '--supervise--new-hrz10-act10'
        writer, env_master, method = gallery.ray_supervise__new_adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-5-7':
        scale = 10
        config.description += '--supervise-roll-hrz10-act10'
        writer, env_master, method = gallery.ray_supervise_roll__adaptive_background__bottleneck(config, mode, scale)


    elif version == 'v6-5-8':
        scale = 10
        config.description += '--supervise-discrete-svo'
        writer, env_master, method = gallery.ray_supervise__adaptive_background__bottleneck_discrete_svo(config, mode, scale)
        # try 改成随机离散分布

    elif version == 'v6-5-9':
        scale = 10
        config.description += '--wo_attention-hrz10-act10'
        writer, env_master, method = gallery.ray_wo_attention__new_adaptive_background__bottleneck(config, mode, scale)

    elif version == 'v6-6-0':
        scale = 10
        config.description += '--isac_recog__new_action'
        writer, env_master, method = gallery.ray_recog__new_action_background__bottleneck(config, mode, scale)
    
    elif version == 'v6-6-1':
        scale = 1
        config.description += '--recog_new_action_woattn'
        writer, env_master, method = gallery.ray_recog__new_action_woattn_background__bottleneck(config, mode, scale)
    
    elif version == 'v6-6-2':
        scale = 1
        config.description += '--recog__dynamic_action'
        writer, env_master, method = gallery.ray_recog__dynamic_action_background__bottleneck(config, mode, scale)
    elif version == 'v6-6-3':
        scale = 1
        config.description += '--recog_woattn__dynamic_action'
        writer, env_master, method = gallery.ray_recog_woattn__dynamic_action_background__bottleneck(config, mode, scale)
        # try 改成随机离散分布
    ################################################################################################
    ##### debug ####################################################################################
    ################################################################################################



    elif version == 'v10':
        scale = 1
        writer, env_master, method = gallery.ray_sac__roundabout__adaptive_background(config, mode, scale)


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
    
