from distutils.log import debug
import rllib
import universe
import ray

import time
import psutil
import torch


class Debug(object): 
    episode = -1
    debug_episode = 0
    time_step = -1 
    debug_time_step = 200
    def __init__(self) -> None:
        pass
    def run_one_episode(self, env, method):
        t1 = time.time()
        env.reset()
        state = [s.to_tensor().unsqueeze(0) for s in env.state]
        t2 = time.time()
        time_select_action = 0.0
        time_env_step = 0.0
        ## add debug 
        self.episode += 1
        if (self.episode < self.debug_episode) : return

        while True:

            if env.config.render:
                env.render()
                # import pdb; pdb.set_trace()

            tt1 = time.time()
            action = method.select_actions(state).cpu().numpy()
            tt2 = time.time()
            experience, done, info = env.step(action)
            tt3 = time.time()
            time_select_action += (tt2-tt1)
            time_env_step += (tt3-tt2)
            state = [s.to_tensor().unsqueeze(0) for s in env.state]
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
    state = [s.to_tensor().unsqueeze(0) for s in env.state]
    t2 = time.time()
    time_select_action = 0.0
    time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()
            # import pdb; pdb.set_trace()

        tt1 = time.time()
        action = method.select_actions(state).cpu().numpy()
        tt2 = time.time()
        experience, done, info = env.step(action)
        tt3 = time.time()
        time_select_action += (tt2-tt1)
        time_env_step += (tt3-tt2)

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
        config.seed += 5
    rllib.basic.setup_seed(config.seed)
    
    import gallery_ma_eval as gallery
    import models_ma
    version = config.version
    if version == 'pseudo':
        raise NotImplementedError
    
    ################################################################################################
    ##### evaluate, recognition, bottleneck ########################################################
    ################################################################################################

    elif version == 'vtrue-0':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        config.description = 'adaptive_bottleneck'
        models_ma.isac__bottleneck__adaptive().update(config)
        env_master = gallery.evaluate_ray_isac_adaptive_character__bottleneck(config, mode, scale)
    
    elif version == 'vtrue-1': 
        if mode != 'evaluate':
            raise NotImplementedError
        import numpy as np
        for svo in np.linspace(0, 1, num=11):
            svo = round(svo,1)
            config.description = 'evaluate' + '--fix_{}__adaptive_bottleneck'.format(svo)
            models_ma.isac__bottleneck__adaptive().update(config)
            env_master = gallery.evaluate_ray_isac_adaptive_character__bottleneck_fix_svo(config,svo,mode)
            env_master.create_tasks(func=run_one_episode)
            ray.get([t.run.remote(n_iters=config.num_episodes) for t in env_master.tasks])
            del env_master
            ray.shutdown()
            ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)
        ray.shutdown()
        return
        # svo = config.svo
        # config.description = 'evaluate' + '--fix_{}__bottleneck'.format(svo)
        # models_ma.isac__bottleneck__adaptive().update(config)
        # env_master = gallery.evaluate_ray_isac_adaptive_character__bottleneck_fix_svo(config,svo,mode)



    elif version == 'v1-4-0':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        config.description = 'RILMthM__bottleneck'
        models_ma.RILMthM__bottleneck().update(config)
        env_master = gallery.evaluate_ray_RILMthM__bottleneck(config, mode, scale)
    
    elif version == 'v1-4-1':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        config.description = 'RILEnvM__bottleneck'
        models_ma.RILEnvM__bottleneck().update(config)
        env_master = gallery.evaluate_ray_RILEnvM__bottleneck(config, mode, scale)
    
    elif version == 'v1-4-2':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        config.description = 'ILEnvM__bottleneck'
        models_ma.ILEnvM__bottleneck().update(config)
        env_master = gallery.evalute_ray_supervise_offline_multiagent__bottleneck(config, mode, scale)

    ################################################################################################
    ##### evaluate, recognition, merge #############################################################
    ################################################################################################
    
    elif version == 'vtrue-2':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        config.description = 'adaptive_merge'
        models_ma.isac_adaptive_character__merge().update(config)
        env_master = gallery.evaluate_ray_isac_adaptive_character__merge(config, mode, scale)
    
    elif version == 'vtrue-3': 
        if mode != 'evaluate':
            raise NotImplementedError
        import numpy as np
        for svo in np.linspace(0, 1, num=11):
            svo = round(svo,1)
            config.description = 'evaluate' + '--fix_{}_adaptive_merge'.format(svo)
            models_ma.isac_adaptive_character__merge().update(config)
            env_master = gallery.evaluate_ray_isac_adaptive_character__merge_fix_svo(config,svo,mode)
            env_master.create_tasks(func=run_one_episode)
            ray.get([t.run.remote(n_iters=config.num_episodes) for t in env_master.tasks])
            del env_master
            ray.shutdown()
            ray.init(num_cpus=psutil.cpu_count(), num_gpus=torch.cuda.device_count(), include_dashboard=False)
        ray.shutdown()
        return


    elif version == 'v2-4-0':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        config.description = 'RILMthM__merge'
        models_ma.RILMthM__merge().update(config)
        env_master = gallery.evaluate_ray_RILMthM__merge(config, mode, scale)
    
    elif version == 'v2-4-1':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        config.description = 'RILEnvM__merge'

        models_ma.RILEnvM__merge().update(config)
        env_master = gallery.evaluate_ray_RILEnvM__merge(config, mode, scale)
    
    elif version == 'v2-4-2':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        config.description = 'ILEnvM__merge'

        models_ma.ILEnvM__merge().update(config)
        env_master = gallery.evalute_ray_supervise_offline_multiagent__merge(config, mode, scale)

    elif version == 'v3-1':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        # scale = 1
        config.description = 'isac_robust_character__bottleneck'

        models_ma.isac_robust_character__bottleneck().update(config)
        env_master = gallery.evaluate_ray_isac_robust_character__bottleneck(config, mode, scale)

    elif version == 'v3-2':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        # scale = 1
        config.description = 'isac_robust_character__merge'

        models_ma.isac_robust_character__merge().update(config)
        env_master = gallery.evaluate_ray_isac_robust_character__merge(config, mode, scale)


    elif version == 'v4-1':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        # scale = 1
        config.description = 'isac_no_character__bottleneck'

        # models_ma.isac_no_character__multi_scenario().update(config)
        models_ma.isac_no_character__bottleneck().update(config)
        env_master = gallery.evaluate_ray_isac_no_character__bottleneck(config, mode, scale)
    
    elif version == 'v4-2':
        if mode != 'evaluate':
            raise NotImplementedError

        scale = 5
        # scale = 1
        config.description = 'isac_no_character__multi_scenario'

        models_ma.isac_no_character__multi_scenario().update(config)
        env_master = gallery.evaluate_ray_isac_no_character__merge(config, mode, scale)

    # ################################################################################################
    # ##### evaluate, training setting ###############################################################
    # ################################################################################################


    # elif version == 'v0-1':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_adaptive_character__bottleneck().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character__bottleneck(config, mode, scale)


    # elif version == 'v0-2':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_adaptive_character__intersection().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character__intersection(config, mode, scale)
    


    # elif version == 'v0-3':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_adaptive_character__merge().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character__merge(config, mode, scale)
    


    # elif version == 'v0-4':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_adaptive_character__roundabout().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character__roundabout(config, mode, scale)
    


    # ################################################################################################
    # ##### evaluate, training setting robust ########################################################
    # ################################################################################################


    # elif version == 'v1-1':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character__bottleneck().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character__bottleneck(config, mode, scale)

    # elif version == 'v1-2':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character__intersection().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character__intersection(config, mode, scale)

    # elif version == 'v1-3':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character__merge().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character__merge(config, mode, scale)

    # elif version == 'v1-4':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character__roundabout().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character__roundabout(config, mode, scale)




    # ################################################################################################
    # ##### evaluate, training setting copo ##########################################################
    # ################################################################################################


    # elif version == 'v2-1':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character_copo__multi_scenario().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character_copo__bottleneck(config, mode, scale)

    # elif version == 'v2-2':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character_copo__multi_scenario().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character_copo__intersection(config, mode, scale)

    # elif version == 'v2-3':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character_copo__multi_scenario().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character_copo__merge(config, mode, scale)

    # elif version == 'v2-4':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character_copo__multi_scenario().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character_copo__roundabout(config, mode, scale)





    # ################################################################################################
    # ##### evaluate, training setting no ############################################################
    # ################################################################################################


    # elif version == 'v3-1':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     # models_ma.isac_no_character__multi_scenario().update(config)
    #     models_ma.isac_no_character__bottleneck().update(config)
    #     env_master = gallery.evaluate_ray_isac_no_character__bottleneck(config, mode, scale)

    # elif version == 'v3-2':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_no_character__multi_scenario().update(config)
    #     env_master = gallery.evaluate_ray_isac_no_character__intersection(config, mode, scale)

    # elif version == 'v3-3':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_no_character__multi_scenario().update(config)
    #     env_master = gallery.evaluate_ray_isac_no_character__merge(config, mode, scale)

    # elif version == 'v3-4':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_no_character__multi_scenario().update(config)
    #     env_master = gallery.evaluate_ray_isac_no_character__roundabout(config, mode, scale)






    # ################################################################################################
    # ##### evaluate, training setting copo adv ######################################################
    # ################################################################################################


    # elif version == 'v4-1':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character_copo__multi_scenario().update(config)  ### pseudo
    #     env_master = gallery.evaluate_ray_isac_robust_character_copo_adv__bottleneck(config, mode, scale)

    # elif version == 'v4-2':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character_copo__multi_scenario().update(config)  ### pseudo
    #     env_master = gallery.evaluate_ray_isac_robust_character_copo_adv__intersection(config, mode, scale)

    # elif version == 'v4-3':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character_copo__multi_scenario().update(config)  ### pseudo
    #     env_master = gallery.evaluate_ray_isac_robust_character_copo_adv__merge(config, mode, scale)

    # elif version == 'v4-4':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 5
    #     # scale = 1
    #     models_ma.isac_robust_character_copo__multi_scenario().update(config)  ### pseudo
    #     env_master = gallery.evaluate_ray_isac_robust_character_copo_adv__roundabout(config, mode, scale)














    # ################################################################################################
    # ##### evaluate, assign character, adaptive #####################################################
    # ################################################################################################

    # elif version == 'v-1-1':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 11
    #     models_ma.isac_adaptive_character__bottleneck().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character_assign__bottleneck(config, mode, scale)

    # elif version == 'v-1-2':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 11
    #     models_ma.isac_adaptive_character__intersection().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character_assign__intersection(config, mode, scale)

    # elif version == 'v-1-3':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 11
    #     # models_ma.isac_adaptive_character__multi_scenario().update(config)
    #     models_ma.isac_adaptive_character__merge().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character_assign__merge(config, mode, scale)

    # elif version == 'v-1-4':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 11
    #     # models_ma.isac_adaptive_character__multi_scenario().update(config)
    #     models_ma.isac_adaptive_character__roundabout().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character_assign__roundabout(config, mode, scale)




    # ################################################################################################
    # ##### evaluate, assign character, robust #######################################################
    # ################################################################################################

    # elif version == 'v-1-5':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 11
    #     models_ma.isac_robust_character__bottleneck().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character_assign__bottleneck(config, mode, scale)

    # elif version == 'v-1-6':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 11
    #     models_ma.isac_robust_character__intersection().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character_assign__intersection(config, mode, scale)

    # elif version == 'v-1-7':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 11
    #     models_ma.isac_robust_character__merge().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character_assign__merge(config, mode, scale)

    # elif version == 'v-1-8':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 11
    #     models_ma.isac_robust_character__roundabout().update(config)
    #     env_master = gallery.evaluate_ray_isac_robust_character_assign__roundabout(config, mode, scale)








    # ################################################################################################
    # ##### evaluate, diversity ######################################################################
    # ################################################################################################



    # elif version == 'v-2-1':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 1
    #     models_ma.isac_adaptive_character__bottleneck__qualitive().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character_diversity__bottleneck(config, mode, scale)





    # ################################################################################################
    # ##### evaluate, social behavior ################################################################
    # ################################################################################################

    # elif version == 'v-3-1':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 1
    #     models_ma.isac_adaptive_character__bottleneck__qualitive().update(config)
    #     # models_ma.isac_adaptive_character__bottleneck().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character__social_behavior__bottleneck(config, mode, scale)


    # elif version == 'v-3-2':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 1
    #     # models_ma.isac_adaptive_character__intersection__qualitive().update(config)
    #     models_ma.isac_adaptive_character__intersection().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character__social_behavior__intersection(config, mode, scale)

    # elif version == 'v-3-3':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 1
    #     # models_ma.isac_adaptive_character__multi_scenario().update(config)
    #     models_ma.isac_adaptive_character__merge().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character__social_behavior__merge(config, mode, scale)

    # elif version == 'v-3-4':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 1
    #     models_ma.isac_adaptive_character__roundabout().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character__social_behavior__roundabout(config, mode, scale)









    # ################################################################################################
    # ##### debug ####################################################################################
    # ################################################################################################


    # elif version == 'v10-1':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 1
    #     models_ma.isac_adaptive_character__bottleneck().update(config)
    #     # models_ma.isac_adaptive_character__multi_scenario().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character_assign__bottleneck(config, mode, scale)

    # elif version == 'v10-2':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 1
    #     models_ma.isac_adaptive_character__intersection().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character_assign__intersection(config, mode, scale)

    # elif version == 'v10-3':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 1
    #     models_ma.isac_adaptive_character__multi_scenario().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character_assign__merge(config, mode, scale)

    # elif version == 'v10-4':
    #     if mode != 'evaluate':
    #         raise NotImplementedError

    #     scale = 1
    #     # models_ma.isac_adaptive_character__multi_scenario().update(config)
    #     models_ma.isac_adaptive_character__roundabout().update(config)
    #     env_master = gallery.evaluate_ray_isac_adaptive_character_assign__roundabout(config, mode, scale)









    else:
        raise NotImplementedError


    try:
        debug = Debug()
        env_master.create_tasks(run_one_episode)
        ray.get([t.run.remote(n_iters=config.num_episodes) for t in env_master.tasks])

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()





if __name__ == '__main__':
    main()
    
