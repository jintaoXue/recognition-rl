import rllib
import universe

import gym
import torch
import time




def run_one_episode(i_episode, config, writer, env, method):
    t1 = time.time()
    env.reset()
    state = [s.to_tensor().unsqueeze(0) for s in env.state]
    t2 = time.time()
    time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()
        
        action = method.select_actions(state).cpu().numpy()
        tt1 = time.time()
        experience, done, info = env.step(action)
        tt2 = time.time()
        time_env_step += (tt2-tt1)

        # experience = rllib.template.Experience(
        #         state=torch.from_numpy(state).float().unsqueeze(0),
        #         next_state=torch.from_numpy(next_state).float().unsqueeze(0),
        #         action=action.cpu(), reward=reward, done=done, info=rllib.basic.Data(**info))
        method.store(experience)
        state = [s.to_tensor().unsqueeze(0) for s in env.state]
        if done:
            break
    
    method.update_parameters()

    env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
    env.writer.add_scalar('time_analysis/step', time_env_step, env.step_reset)
    return


def main():
    config = rllib.basic.YamlConfig()
    from config.args import generate_args
    args = generate_args()
    config.update(args)

    mode = 'train'
    if config.evaluate == True:
        mode = 'evaluate'
        config.seed += 1
    rllib.basic.setup_seed(config.seed)
    
    import gallery_ma as gallery
    import models_ma
    version = config.version
    if version == 'v1-1':
        writer, env, method = gallery.isac_no_character__bottleneck(config, mode)

    elif version == 'v2-1':
        writer, env, method = gallery.isac_robust_character__bottleneck(config, mode)

    elif version == 'v3-1':
        if mode == 'evaluate':
            config.method = models_ma.isac_adaptive_character__bottleneck.method
            config.model_dir = models_ma.isac_adaptive_character__bottleneck.model_dir
            config.model_num = models_ma.isac_adaptive_character__bottleneck.model_num
        writer, env, method = gallery.isac_adaptive_character__bottleneck(config, mode)

    



    ### evaluate, assign character
    elif version == 'v-1-1':
        if mode != 'evaluate':
            raise NotImplementedError

        config.method = models_ma.isac_adaptive_character__bottleneck.method
        config.model_dir = models_ma.isac_adaptive_character__bottleneck.model_dir
        config.model_num = models_ma.isac_adaptive_character__bottleneck.model_num
        writer, env, method = gallery.evaluate_isac_adaptive_character__bottleneck(config, mode)









    elif version == 'v10':
        writer, env, method = gallery.debug__isac_adaptive_character__merge(config, mode)
    elif version == 'v10-1':
        writer, env, method = gallery.debug__isac_adaptive_character__roundabout(config, mode)
    elif version == 'v10-2':
        writer, env, method = gallery.debug__isac_adaptive_character__intersection(config, mode)
    
    else:
        raise NotImplementedError

    try:
        for i_episode in range(10000):
            run_one_episode(i_episode, config, writer, env, method)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
    
