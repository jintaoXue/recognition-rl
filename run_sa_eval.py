
import rllib
import universe
import ray

import time
import psutil
import torch



class Debug(object): 
    episode = -1
    debug_episode = 2
    time_step = -1 
    debug_time_step = 0
    def __init__(self) -> None:
        pass
    def run_one_episode(self, env, method):
        t1 = time.time()
        env.reset()
        state = env.state[0].to_tensor().unsqueeze(0)
        t2 = time.time()
        time_env_step = 0.0
        time_select_action = 0.0
        ## add debug 
        self.episode += 1
        if (self.episode < self.debug_episode) : return

        while True:

            if env.config.render:
                env.render()
                # import pdb; pdb.set_trace()

            tt1 = time.time()
            action = method.select_action(state).cpu().numpy()
            tt2 = time.time()
            experience, done, info = env.step(action)
            tt3 = time.time()
            time_select_action += (tt2-tt1)
            time_env_step += (tt3-tt2)
            state = env.state[0].to_tensor().unsqueeze(0)
            self.time_step += 1
            if (self.time_step >= self.debug_time_step) :breakpoint()
            if done:
                self.time_step = -1
                break
        
        env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
        env.writer.add_scalar('time_analysis/select_action', time_select_action, env.step_reset)
        env.writer.add_scalar('time_analysis/step', time_env_step, env.step_reset)
        return


def run_one_episode(env, method):
    t1 = time.time()
    env.reset()
    state = env.state[0].to_tensor().unsqueeze(0)
    t2 = time.time()
    time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()

        action = method.select_action(state).cpu().numpy()
        tt1 = time.time()
        experience, done, info = env.step(action)
        tt2 = time.time()
        # time_env_step += (tt2-tt1)

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
    
    import gallery_sa_eval as gallery
    import models_sa, models_ma
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError
    
    ################################################################################################
    ##### evaluate, training setting ###############################################################
    ################################################################################################
    elif version == 'v0-1':
        if mode != 'evaluate':
            raise NotImplementedError
        scale = 1
        models_sa.sac__bottleneck__idm().update(config)
        env_master = gallery.evaluate_ray_sac__bottleneck__idm_background(config, mode)


    ################################################################################################
    ##### evaluate, adaptive background without mismatch ###########################################
    ################################################################################################
    elif version == 'v-1-1':
        if mode != 'evaluate':
            raise NotImplementedError
        
        scale = 5
        models_sa.sac__bottleneck__adaptive_adv_background().update(config)
        env_master = gallery.evaluate_ray_sac__bottleneck__adaptive_background(config, mode, scale)

    elif version == 'v-1-2':
        if mode != 'evaluate':
            raise NotImplementedError
        
        scale = 1
        models_sa.ppo__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate_ray_ppo__bottleneck__adaptive_background(config, mode, scale)

    elif version == 'v-1-3':
        if mode != 'evaluate':
            raise NotImplementedError
        
        scale = 5
        models_sa.sac__bottleneck__adaptive_adv_background().update(config)
        env_master = gallery.evaluate_ray_sac__bottleneck__multi_background(config, mode, scale)


    elif version == 'v-1-4':
        if mode != 'evaluate':
            raise NotImplementedError
        
        scale = 1
        models_sa.ppo__bottleneck__idm().update(config)
        env_master = gallery.evaluate_ray_ppo__bottleneck__idm_background(config, mode, scale)


    ################################################################################################
    ##### evaluate, full stack, bottleneck #########################################################
    ################################################################################################


    elif version == 'v1-1':  ### idm
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--idm'
        models_sa.sac__bottleneck__idm().update(config)
        env_master = gallery.evaluate__full_stack_background__bottleneck(config, mode)

    elif version == 'v1-2':  ### no_character
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--no_character'
        models_sa.sac__bottleneck__no_character().update(config)
        env_master = gallery.evaluate__full_stack_background__bottleneck(config, mode)

    elif version == 'v1-3':  ### robust_copo
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--robust_copo'
        models_sa.sac__bottleneck__robust_copo().update(config)
        env_master = gallery.evaluate__full_stack_background__bottleneck(config, mode)

    elif version == 'v1-4':  ### explicit_adv
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--explicit_adv'
        models_sa.sac__bottleneck__explicit_adv().update(config)
        env_master = gallery.evaluate__full_stack_background__bottleneck(config, mode)

    elif version == 'v1-5':  ### adaptive
        if mode != 'evaluate':
            raise NotImplementedError
        
        config.description += '--adaptive'
        models_sa.sac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__full_stack_background__bottleneck(config, mode)


    elif version == 'v1-6':  ### duplicity
        if mode != 'evaluate':
            raise NotImplementedError
        
        config.description += '--duplicity'
        models_sa.sac__bottleneck__duplicity().update(config)
        env_master = gallery.evaluate__full_stack_background__bottleneck(config, mode)




    ################################################################################################
    ##### evaluate, full stack, intersection #######################################################
    ################################################################################################


    elif version == 'v2-1':  ### idm
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--idm'
        models_sa.sac__intersection__idm().update(config)
        env_master = gallery.evaluate__full_stack_background__intersection(config, mode)

    elif version == 'v2-2':  ### no_character
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--no_character'
        models_sa.sac__intersection__no_character().update(config)
        env_master = gallery.evaluate__full_stack_background__intersection(config, mode)

    elif version == 'v2-3':  ### robust_copo
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--robust_copo'
        models_sa.sac__intersection__robust_copo().update(config)
        env_master = gallery.evaluate__full_stack_background__intersection(config, mode)

    elif version == 'v2-4':  ### explicit_adv
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--explicit_adv'
        models_sa.sac__intersection__explicit_adv().update(config)
        env_master = gallery.evaluate__full_stack_background__intersection(config, mode)

    elif version == 'v2-5':  ### adaptive
        if mode != 'evaluate':
            raise NotImplementedError
        
        config.description += '--adaptive'
        models_sa.sac__intersection__adaptive().update(config)
        env_master = gallery.evaluate__full_stack_background__intersection(config, mode)


    elif version == 'v2-6':  ### duplicity
        if mode != 'evaluate':
            raise NotImplementedError
        
        config.description += '--duplicity'
        models_sa.sac__intersection__duplicity().update(config)
        env_master = gallery.evaluate__full_stack_background__intersection(config, mode)






    ################################################################################################
    ##### evaluate, full stack, merge ##############################################################
    ################################################################################################


    elif version == 'v3-1':  ### idm
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--idm'
        models_sa.sac__merge__idm().update(config)
        env_master = gallery.evaluate__full_stack_background__merge(config, mode)

    elif version == 'v3-2':  ### no_character
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--no_character'
        models_sa.sac__merge__no_character().update(config)
        env_master = gallery.evaluate__full_stack_background__merge(config, mode)

    elif version == 'v3-3':  ### robust_copo
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--robust_copo'
        models_sa.sac__merge__robust_copo().update(config)
        env_master = gallery.evaluate__full_stack_background__merge(config, mode)

    elif version == 'v3-4':  ### explicit_adv
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--explicit_adv'
        models_sa.sac__merge__explicit_adv().update(config)
        env_master = gallery.evaluate__full_stack_background__merge(config, mode)

    elif version == 'v3-5':  ### adaptive
        if mode != 'evaluate':
            raise NotImplementedError
        
        config.description += '--adaptive'
        models_sa.sac__merge__adaptive().update(config)
        env_master = gallery.evaluate__full_stack_background__merge(config, mode)


    elif version == 'v3-6':  ### duplicity
        if mode != 'evaluate':
            raise NotImplementedError
        
        config.description += '--duplicity'
        models_sa.sac__merge__duplicity().update(config)
        env_master = gallery.evaluate__full_stack_background__merge(config, mode)






    ################################################################################################
    ##### evaluate, full stack, roundabout #########################################################
    ################################################################################################


    elif version == 'v4-1':  ### idm
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--idm'
        models_sa.sac__roundabout__idm().update(config)
        env_master = gallery.evaluate__full_stack_background__roundabout(config, mode)

    elif version == 'v4-2':  ### no_character
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--no_character'
        models_sa.sac__roundabout__no_character().update(config)
        env_master = gallery.evaluate__full_stack_background__roundabout(config, mode)

    elif version == 'v4-3':  ### robust_copo
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--robust_copo'
        models_sa.sac__roundabout__robust_copo().update(config)
        env_master = gallery.evaluate__full_stack_background__roundabout(config, mode)

    elif version == 'v4-4':  ### explicit_adv
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--explicit_adv'
        models_sa.sac__roundabout__explicit_adv().update(config)
        env_master = gallery.evaluate__full_stack_background__roundabout(config, mode)

    elif version == 'v4-5':  ### adaptive
        if mode != 'evaluate':
            raise NotImplementedError
        
        config.description += '--adaptive'
        models_sa.sac__roundabout__adaptive().update(config)
        env_master = gallery.evaluate__full_stack_background__roundabout(config, mode)


    elif version == 'v4-6':  ### duplicity
        if mode != 'evaluate':
            raise NotImplementedError
        
        config.description += '--duplicity'
        models_sa.sac__roundabout__duplicity().update(config)
        env_master = gallery.evaluate__full_stack_background__roundabout(config, mode)






    ################################################################################################
    ##### evaluate ----->>> idm copo ours flow backgrounds, bottleneck #############################
    ################################################################################################


    elif version == 'v5-1':  ### robust_copo
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--roboust_copo'
        models_sa.isac__bottleneck__robust_copo().update(config)
        env_master = gallery.evaluate__isac_roubust__four_background__bottleneck(config, mode)

    elif version == 'v5-2':  ### adaptive + four backgrounds
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--adaptive'
        models_sa.isac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__isac_adaptive__four_background__bottleneck(config, mode)
    
    elif version == 'v5-2-1':  ### adaptive + one backgrounds 
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--adaptive'
        models_sa.isac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__isac_adaptive__adaptive_background__bottleneck(config, mode)
    
    elif version == 'v5-2-2':  ### adaptive + one backgrounds + down sample
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--adaptive-downsample'
        models_sa.isac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__isac__adaptive_background_downsample_bottleneck(config, mode)
    
    elif version == 'v5-2-3':  ### adaptive + one backgrounds + same other svo
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--adaptive-othersvo_0.8'
        models_sa.isac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__isac_adaptive__adaptive_background__bottleneck(config, mode)
    
    elif version == 'v5-3':  ### 
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--sac__four_background__bottleneck'
        models_sa.sac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__sac__four_background__bottleneck(config, mode)


    ################################################################################################
    ##### evaluate, character recognition ##########################################################
    ################################################################################################


    elif version == 'v6-0-1':  ### idm
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--recog_idm'
        models_sa.isac_recog__bottleneck__idm().update(config)
        env_master = gallery.evaluate__isac_recog__idm_background__bottleneck(config, mode)

    elif version == 'v6-0-2':  ### adaptive
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--isac_recog__adaptive_background__bottleneck'
        models_sa.isac_recog__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__isac_recog__adaptive_background__bottleneck(config, mode)

    elif version == 'v6-2':  ### no character + four backgrounds
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--isac_recog__four_background__bottleneck'
        models_sa.isac_recog__bottleneck__no_character().update(config)
        env_master = gallery.evaluate__isac_recog__four_background__bottleneck(config, mode)

    elif version == 'v6-3':  ### robust + four backgrounds
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--isac_recog__four_background__bottleneck'
        models_sa.isac_recog__bottleneck__robust().update(config)
        env_master = gallery.evaluate__isac_recog__four_background__bottleneck(config, mode)

    elif version == 'v6-4':  ### adaptive + four backgrounds
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--isac_recog__four_background__bottleneck'
        models_sa.isac_recog__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__isac_recog__four_background__bottleneck(config, mode)
    
    elif version == 'v6-4-1':  ### adaptive + one backgrounds
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--isac_recog_hr10act1__adaptive_background__bottleneck'
        models_sa.isac_recog__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__isac_recog__one_background__bottleneck(config, mode)

    elif version == 'v6-4-2':  ### adaptive + one backgrounds + downsample
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--recog_hr10act1__adaptive_background_downsample__bottleneck'
        models_sa.isac_recog__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__recog__one_background_downsample__bottleneck(config, mode)

    elif version == 'v6-4-3':  ### adaptive + four backgrounds
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--isac_recog_hr10act1__four_background__bottleneck'
        models_sa.isac_recog__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__isac_recog__four_background__bottleneck(config, mode)

    elif version == 'v6-4-4':  ### adaptive + one backgrounds + random svo
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--isac_recog_hr10act1__adaptive_background__bottleneck_random_svo'
        models_sa.isac_recog__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__recog_random_svo_one_background__bottleneck(config, mode)

    elif version == 'v6-4-5':  ### adaptive + one backgrounds + fix other svo + without attn
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--isac_recog_hr10act1__bottleneck_woattn'
        models_sa.isac_recog_woattn__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__recog_woattn__one_background__bottleneck(config, mode)

    elif version == 'v6-5':  ### adaptive + supervise + four backgrounds
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--supervise__four_background__bottleneck-hr30act10'
        models_sa.supervise__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__supervise__four_background__bottleneck(config, mode)

    ################################################################################################
    ##### evaluate, fixed character ################################################################
    ################################################################################################

    elif version == 'v7-0':  ### ours 
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--fix_0.8_0.5__four_background__bottleneck'
        models_sa.isac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__fix_svo__four_background__bottleneck(config, mode)

    elif version == 'v7-1':  ### ours 
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--fix_0.8_0.5__two_background__bottleneck'
        models_sa.isac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate__fix_svo__two_background__bottleneck(config, 0.8, 0.5,mode)

    elif version == 'v7-2-0': 
        if mode != 'evaluate':
            raise NotImplementedError
        import numpy as np
        for ego_svo in np.linspace(0, 0, num=1):
            for other_svo in np.linspace(0, 1, num=11):
                config.description = 'evaluate' + '--fix_{}_{}__one_background__bottleneck'.format(ego_svo, other_svo)
                breakpoint()
                models_sa.isac__bottleneck__adaptive().update(config)
                env_master = gallery.evaluate__fix_svo__new_one_background__bottleneck(config, ego_svo, other_svo,mode)
                env_master.create_tasks(func=run_one_episode)
                ray.get([t.run.remote(n_iters=200) for t in env_master.tasks])
                del env_master
                ray.shutdown()
                ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)
        for ego_svo in np.linspace(0, 0, num=1):
            for other_svo in np.linspace(0, 1, num=11):
                config.description = 'evaluate' + '--recog_fix_{}_{}__one_background__bottleneck'.format(ego_svo, other_svo)
                models_sa.isac_recog__bottleneck__adaptive().update(config)
                env_master = gallery.evaluate__recog_fix_svo__new_one_background__bottleneck(config, ego_svo, other_svo,mode)
                env_master.create_tasks(func=run_one_episode)
                ray.get([t.run.remote(n_iters=200) for t in env_master.tasks])
                del env_master
                ray.shutdown()
                ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)
    else:
        raise NotImplementedError



    
    try:
        # debug = Debug()
        env_master.create_tasks(func=run_one_episode)

        ray.get([t.run.remote(n_iters=200) for t in env_master.tasks])


    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


# def _main(ego_svo, other_svo) :
def _main() :
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
    
    import gallery_sa_eval as gallery
    import models_sa, models_ma
    version = config.version
    if version == 'v6-4-1':
        if mode != 'evaluate':
            raise NotImplementedError
        # debug = Debug()
        eval_end_num = 211000
        interval = 2000
        for num in range(0, 100):
            model_num = eval_end_num - num*interval
            # model_num = 100000
            config.description = 'recog_hr10act1__adaptive_background__bottleneck'
            models_sa.recog_rl__bottleneck__adaptive__given_number().update(config, model_num)
            env_master = gallery.evaluate__isac_recog__one_background__bottleneck(config, mode)
            env_master.create_tasks(func=run_one_episode)
            ray.get([t.run.remote(n_iters=200) for t in env_master.tasks])
            del env_master
            ray.shutdown()
            ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)
    
    if version == 'v6-4-2':
        if mode != 'evaluate':
            raise NotImplementedError
        # debug = Debug()
        eval_end_num = 211000
        interval = 2000
        for num in range(0, 100):
            model_num = eval_end_num - num*interval
            # model_num = 100000
            config.description = 'recog_hr10act1__adaptive_background_downsample__bottleneck'
            models_sa.recog_rl__bottleneck__adaptive__given_number().update(config, model_num)
            env_master = gallery.evaluate__recog__one_background_downsample__bottleneck(config, mode)
            env_master.create_tasks(func=run_one_episode)
            ray.get([t.run.remote(n_iters=200) for t in env_master.tasks])
            del env_master
            ray.shutdown()
            ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)

    if version == 'v6-5-0':
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--supervise__four_background__bottleneck-hr30act10'
        models_sa.supervise__bottleneck__adaptive__given_number().update(config, model_num)
        env_master = gallery.evaluate__supervise__four_background__bottleneck(config, mode)

    if version == 'v6-5-1':
        if mode != 'evaluate':
            raise NotImplementedError

        config.description += '--supervise__four_background__bottleneck-hr10act10'
        models_sa.supervise__bottleneck__adaptive__given_number().update(config, model_num)
        env_master = gallery.evaluate__supervise__four_background__bottleneck(config, mode)
    
    if version == 'v6-5-2':
        if mode != 'evaluate':
            raise NotImplementedError
        # debug = Debug()
        eval_end_num = 2250000
        interval = 10000
        for num in range(0, 20):
            model_num = eval_end_num - num*interval
            config.description = '--supervise-sampling__four_background__bottleneck-hr10act10'
            models_sa.supervise_sampling__bottleneck__adaptive__given_number().update(config, model_num)
            env_master = gallery.evaluate__supervise__four_background__bottleneck(config, mode)
            try:
                env_master.create_tasks(func=run_one_episode)
                ray.get([t.run.remote(n_iters=200) for t in env_master.tasks])
            except Exception as e:
                import traceback
                traceback.print_exc()
    

    # if version == 'v7-2':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     config.description = 'evaluate' + '--fix_{}_{}__two_background__bottleneck'.format(0.1*ego_svo, 0.1*other_svo)
    #     models_sa.isac__bottleneck__adaptive().update(config)
    #     env_master = gallery.evaluate__fix_svo__two_background__bottleneck(config, 0.1*ego_svo, 0.1*other_svo,mode)
    

if __name__ == '__main__':
    # for ego_svo in range(0, 11):
    #     for other_svo in range(0, 11):
    #         main(ego_svo, other_svo)
    # eval_end_num = 2990000
    # for num in range(0, 20):
    #     model_num = eval_end_num - num*20000
    # _main()
    main()
    
