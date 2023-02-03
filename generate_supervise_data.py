
import rllib
import ray

import os
import time
import tqdm
import pickle
import psutil
import torch
import universe
from typing import Tuple
from gallery_sa import get_sac__new_bottleneck__adaptive_character_config

# def init(config, mode, Env, Method) -> universe.EnvMaster_v1:
#     repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
#     config.set('github_repos', repos)

#     model_name = Method.__name__ + '-' + Env.__name__
#     writer_cls = rllib.basic.PseudoWriter
#     writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)

#     from core import method_evaluate
#     if config.method == 'sac':
#         Method = method_evaluate.EvaluateSAC
#     elif config.method == 'ppo':
#         Method = method_evaluate.EvaluatePPO
#     elif config.method == 'IndependentSAC_v0':
#         Method = method_evaluate.EvaluateIndependentSAC
#     elif config.method == 'IndependentSAC_recog':
#         Method = method_evaluate.EvaluateSACRecog
#     elif config.method == 'IndependentSAC_supervise' or config.method == 'IndependentSACsupervise':
#         Method = method_evaluate.EvaluateSACsupervise
#     else:
#         raise NotImplementedError
    
#     from universe import EnvMaster_v1 as EnvMaster
#     env_master = EnvMaster(config, writer, env_cls=Env, method_cls=Method)
#     return env_master

def init(config, mode, Env, Method) -> Tuple[rllib.basic.Writer, universe.EnvMaster_v0, rllib.template.Method]:
    repos = ['~/github/zdk/rl-lib', '~/github/ali/universe', '~/github/zdk/recognition-rl']
    config.set('github_repos', repos)

    model_name = Method.__name__ + '-' + Env.__name__
    writer_cls = rllib.basic.PseudoWriter
    writer = rllib.basic.create_dir(config, model_name, mode=mode, writer_cls=writer_cls)
    
    from universe import EnvMaster_v0 as EnvMaster
    env_master = EnvMaster(config, writer, env_cls=Env)

    config_method = env_master.config_methods[0]
    Method = ray.remote(num_cpus=config_method.num_cpus, num_gpus=config_method.num_gpus)(Method).remote
    method = Method(config_method, writer)
    method.reset_writer.remote()

    return writer, env_master, method

def run_one_episode_single_agent(env, method):
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
        dir_path = f'./results/data_offline/{env.config.scenario_name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, str(ray.get(method._get_buffer_len.remote())-1) + '.txt')
        # print('file path: ', file_path, env.scenario.scenario_randomization.num_vehicles)
        with open(file_path, 'wb') as f:
            pickle.dump(experience, f)
        print('buffer length: {}, safe txt'.format(ray.get(method._get_buffer_len.remote())))


        # print('\n safe experience')
        state = env.state[0].to_tensor().unsqueeze(0)
        if done:
            break
    
    env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
    env.writer.add_scalar('time_analysis/step', time_env_step, env.step_reset)
    return

###################run########################

def run_one_episode_single_agent(env, method):
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
        dir_path = f'./results/data_offline/{env.config.scenario_name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, str(ray.get(method._get_buffer_len.remote())-1) + '.txt')
        # print('file path: ', file_path, env.scenario.scenario_randomization.num_vehicles)
        with open(file_path, 'wb') as f:
            pickle.dump(experience, f)
        print('buffer length: {}, safe txt'.format(ray.get(method._get_buffer_len.remote())))


        # print('\n safe experience')
        state = env.state[0].to_tensor().unsqueeze(0)
        if done:
            break
    
    env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
    env.writer.add_scalar('time_analysis/step', time_env_step, env.step_reset)
    return

def run_one_episode_multi_agent(env, method):
    # t1 = time.time()
    env.reset()
    state = [s.to_tensor().unsqueeze(0) for s in env.state]
    # t2 = time.time()
    # time_select_action = 0.0
    # time_env_step = 0.0
    while True:
        if env.config.render:
            env.render()

        # tt1 = time.time()
        action = ray.get(method.select_actions.remote(state)).cpu().numpy()
        # tt2 = time.time()
        experience, done, info = env.step(action)
        tt3 = time.time()
        # time_select_action += (tt2-tt1)
        # time_env_step += (tt3-tt2)

        method.store.remote(experience, index=env.env_index)

        dir_path = f'./results/data_offline/{env.config.scenario_name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, str(ray.get(method._get_buffer_len.remote())-1) + '.txt')
        # print('file path: ', file_path, env.scenario.scenario_randomization.num_vehicles)
        with open(file_path, 'wb') as f:
            pickle.dump(experience, f)
        print('buffer length: {}, safe txt'.format(ray.get(method._get_buffer_len.remote())))

        state = [s.to_tensor().unsqueeze(0) for s in env.state]
        if done:
            break
    
    # env.writer.add_scalar('time_analysis/reset', t2-t1, env.step_reset)
    # env.writer.add_scalar('time_analysis/select_action', time_select_action, env.step_reset)
    # env.writer.add_scalar('time_analysis/step', time_env_step, env.step_reset)
    return





###################gallery#################
def generate__supervise_data__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveSingleAgent as Env
    #todo
    from core.method_evaluate import EvaluateIndependentSAC as Method
    ### env param
    from config.bottleneck import config_env__neural_background_same_other_svo as config_bottleneck
    # from utils.topology_map import TopologyMapSampled

    # config_bottleneck.set('topology_map', TopologyMapSampled)
    config_bottleneck.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))
    
    config.set('envs', [
        config_bottleneck
    ] *scale)

    ### method param
    from config.method import config_sac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

def ray_RILMthM__bottleneck(config, mode='train', scale=1):
    from universe import EnvInteractiveMultiAgent as Env
    from core.method_evaluate import EvaluateIndependentSAC as Method
    
    ### env param
    from config.bottleneck import config_env as config_bottleneck
    # config_bottleneck.set('config_neural_policy', get_sac__new_bottleneck__adaptive_character_config(config))

    config.set('envs', [
        config_bottleneck,
    ] *scale)

    ### method param
    from config.method import config_isac__adaptive_character as config_method
    config.set('methods', [config_method])

    return init(config, mode, Env, Method)

if __name__ == "__main__":
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

    version = config.version
    
    if version == 'v6-5-6':  ### generate supervise data 
        config.description += '--supervise__generate_data__bottleneck-hr10act1'
        models_sa.isac__bottleneck__adaptive().update(config)
        writer, env_master, method= generate__supervise_data__bottleneck(config, mode)

    if version == 'v1-4':  ### muiltiagent supervise learning 
        config.description += '--supervise__generate_data__bottleneck-hr10act1'
        models_ma.isac__bottleneck__adaptive().update(config)
        writer, env_master, method= ray_RILMthM__bottleneck(config, mode)

    try:
        env_master.create_tasks(method, func=run_one_episode_multi_agent)

        for i_episode in range(10000):
            total_steps = ray.get([t.run.remote() for t in env_master.tasks])
            print('update episode i_episode: ', i_episode)
            # ray.get(method.update_parameters_.remote(i_episode, n_iters=sum(total_steps)))
        

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


