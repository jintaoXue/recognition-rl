import rllib
import universe

import time
import torch



def run_one_episode(i_episode, config, writer, env, method):
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
        time_env_step += (tt2-tt1)

        method.store(experience)
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

    mode = 'train'
    if config.evaluate == True:
        mode = 'evaluate'
        config.seed += 1
    rllib.basic.setup_seed(config.seed)
    
    import gallery_sa as gallery
    version = config.version
    if version == 'v1':
        if mode == 'evaluate':
            config.method = 'sac'
            config.model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-08-29-03:01:05----ray_sac__bottleneck--batch-128/saved_models_method'
            config.model_num = 348600
        writer, env, method = gallery.sac__bottleneck(config, mode)

    else:
        raise NotImplementedError
    
    try:
        for i_episode in range(10000):
            run_one_episode(i_episode, config, writer, env, method)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
    
