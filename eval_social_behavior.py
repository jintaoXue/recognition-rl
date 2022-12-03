import rllib
import universe
from universe.carla.dataset import DatasetReplay

import os

import time


def run_one_episode(env):
    print('\n\n\n\n')
    t1 = time.time()
    env.reset()
    t2 = time.time()
    if env.config.scenario_name == 'bottleneck':
        env.scenario.boundary = rllib.basic.BaseData(x_min=-100, y_min=180, x_max=100, y_max=220)
    while True:
        tt1 = time.time()
        if env.config.render:
            env.render()
        tt2 = time.time()
        
        action = env.action_space.sample()
        experience, done, info = env.step(action)
        print('time render: ', tt2-tt1)
        if done:
            break
    return


"""
Bottleneck:
- Queueing
- Cutting in
- Yielding

Intersection:
- Yielding
- Rushing
- Creeping

Merge:

Roundabout:
- Queueing
- Cutting in
- Yielding
- Bypassing
"""



if __name__ == "__main__":
    config = rllib.basic.YamlConfig()
    from config.args import generate_args
    args = generate_args()
    config.update(args)

    version = config.version
    if version == 'pseudo':
        raise NotImplementedError
    
    # elif version == 'v1':  ### bottleneck
    #     dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent_v1-Evaluate/2022-09-13-11:46:07----social-behavior-bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'
    # elif version == 'v2':  ### intersection
    #     dataset_dir = '/home/caichicken/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent_v1-Evaluate/2022-09-21-10:52:58----evaluate_ray_isac_adaptive_character__social_behavior__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-422600-evaluate/output/env0_intersection'
    # else:
    #     raise NotImplementedError
    dataset_dir = config.dataset
    dataset_dir = os.path.expanduser(dataset_dir)
    num_cases = len(os.listdir(dataset_dir)) -1

    Env = universe.EnvReplayMultiAgent

    config_env = rllib.basic.YamlConfig.load(f'{dataset_dir}/config.txt')
    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)
    config_env.set('render_save', config.render_save)

    config_env.set('mode', universe.AgentMode.replay)
    config_env.set('case_ids', [os.path.join(dataset_dir, f'{i}.txt') for i in range(config.num_episodes, config.num_episodes + num_cases)])
    config_env.set('dataset_cls', DatasetReplay)
    config_env.set('recorder_cls', universe.PseudoRecorder)

    model_name = 'PseudoMethod' + '-' + Env.__name__
    writer = rllib.basic.create_dir(config, model_name, mode='train')

    config_env.set('path_pack', config.path_pack)
    config_env.set('dataset_name', config.dataset_name)
    
    env = Env(config.env, writer, env_index=0)
    env.step_reset = config.num_episodes - 1
    for i in range(num_cases):
        run_one_episode(env)
