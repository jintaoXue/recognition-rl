import rllib
import universe
from universe.common import ColorLib
from universe.carla.dataset import DatasetReplay

import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d


z_lim = 110
z_start = 60.0


def run_one_episode(env: universe.EnvInteractiveSingleAgent, ax, ax_3d):
    
    env.reset()
    if env.config.scenario_name == 'bottleneck':
        env.scenario.boundary = rllib.basic.BaseData(x_min=-100, y_min=180, x_max=100, y_max=220)

        topology_map_cls = env.config.get('topology_map', universe.common.TopologyMap)
        centerline, sideline = env.scenario.clip_topology(env.scenario.topology_map.centerline, env.scenario.topology_map.sideline)
        env.scenario.topology_map = topology_map_cls(centerline, sideline)

    
    if env.step_reset == 0:
        env.scenario.render(ax)
        ax.invert_xaxis()
        ax.tick_params(left=False, labelleft=False, right=False, labelright=False, top=False, labeltop=False, bottom=False, labelbottom=False)
        plt.pause(0.00001)

        margin = 100
        if env.config.scenario_name == 'bottleneck':
            ax_3d.set_xlim3d(env.scenario.boundary.x_min -margin, env.scenario.boundary.x_max +margin)
            ax_3d.set_ylim3d(env.scenario.boundary.y_min -margin/2, env.scenario.boundary.y_max +margin/2)
            ax_3d.set_zlim3d(0, z_lim +margin)
        else:
            raise NotImplementedError(f'unkown scenario: {env.config.scenario_name}')

        alpha = 0.3
        for segment in env.scenario.topology_map.centerline:
            ax_3d.plot(segment[:,0], segment[:,1], [z_start]*segment.shape[0], '-', color=ColorLib.normal(ColorLib.grey), linewidth=1.0, alpha=alpha)

        for segment in env.scenario.topology_map.sideline:
            ax_3d.plot(segment[:,0], segment[:,1], [z_start]*segment.shape[0], '-', color=ColorLib.normal(ColorLib.dim_grey), linewidth=2.0, alpha=alpha)

        set_axes_equal(ax_3d)
        ax_3d.invert_xaxis()
        # ax_3d.tick_params(left=False, labelleft=False, right=False, labelright=False, top=False, labeltop=False, bottom=False, labelbottom=False)
        # ax_3d.set_xticks([])
        # ax_3d.set_yticks([])
        # ax_3d.set_zticks([])
        plt.pause(0.00001)

    traj = [env.agents_master.vehicles_neural[0].get_state()]
    while True:
        if env.config.render and env.time_step == 0:
            env.render()
        
        action = env.action_space.sample()
        experience, done, info = env.step(action)

        vehicle = env.agents_master.vehicles_neural[0]
        if vehicle.vi == 0:
            traj.append(vehicle.get_state())

        if done:
            break
    return traj


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    # ax.set_box_aspect(np.array([x_range,y_range,z_range]) *2)
    ax.set_box_aspect(np.array([y_range,y_range,z_range]) *2)




if __name__ == "__main__":
    config = rllib.basic.YamlConfig()
    from config.args import generate_args
    args = generate_args()
    config.update(args)

    version = config.version
    if version == 'pseudo':
        raise NotImplementedError
    

    elif version == 'v1':  ### bottleneck
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-17:16:13----diversity-bottleneck--data0--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'
        
        ### data0
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-18:37:57----diversity-bottleneck--data0--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'
    
        ### data1
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-18:40:24----diversity-bottleneck--data1--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'
    
        ### data2
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-18:42:08----diversity-bottleneck--data2--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'

        ### data3
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-18:44:58----diversity-bottleneck--data3--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'

        ### data4
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-18:47:02----diversity-bottleneck--data4--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'
        
        ### data4--20-vehicle
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-18:50:12----diversity-bottleneck--data4--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'

        ### data3--20-vehicle
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-18:53:35----diversity-bottleneck--data3--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'
        
        ### data2--20-vehicle
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-18:57:10----diversity-bottleneck--data2--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'
        
        ### data1--20-vehicle
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-18:59:58----diversity-bottleneck--data1--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'
        ### data0--20-vehicle
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-19:02:56----diversity-bottleneck--data0--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate/output/env0_bottleneck'

        ### data5
        dataset_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-19:08:24----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-5-evaluate/output/env0_bottleneck'

        exp_name = '2022-09-13-19:08:24----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-5-evaluate'
        exp_name = '2022-09-13-19:10:51----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-6-evaluate'
        exp_name = '2022-09-13-19:13:06----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-7-evaluate'
        exp_name = '2022-09-13-19:15:57----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-8-evaluate'
        # exp_name = '2022-09-13-19:18:01----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-9-evaluate'
        # exp_name = '2022-09-13-19:20:12----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-10-evaluate'

        exp_name = '2022-09-13-19:31:46----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-11-evaluate'
        exp_name = '2022-09-13-19:32:26----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-12-evaluate'
        exp_name = '2022-09-13-19:32:35----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-13-evaluate'
        exp_name = '2022-09-13-19:32:57----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-14-evaluate'

        exp_name = '2022-09-13-19:40:04----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-15-evaluate'
        exp_name = '2022-09-13-19:37:06----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-16-evaluate'
        exp_name = '2022-09-13-19:37:10----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-17-evaluate'
        exp_name = '2022-09-13-19:37:14----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-18-evaluate'

        exp_name = '2022-09-13-20:57:29----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-14-evaluate'
        exp_name = '2022-09-13-22:06:08----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-14-evaluate'
        exp_name = '2022-09-13-22:10:51----diversity-bottleneck--20-vehicle--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800--data-15-evaluate'

        dataset_dir = f'~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/{exp_name}/output/env0_bottleneck'
        config.description += f'--from-{exp_name}'


    elif version == 'v2':  ### intersection
        dataset_dir = ''
    else:
        raise NotImplementedError
    

    dataset_dir = os.path.expanduser(dataset_dir)
    num_cases = len(os.listdir(dataset_dir)) -1
    num_cases = 11
    # num_cases = 2

    Env = universe.EnvReplayMultiAgent

    config_env = rllib.basic.YamlConfig.load(f'{dataset_dir}/config.txt')
    config.set('env', config_env)
    config_env.set('seed', config.seed)
    config_env.set('render', config.render)
    config_env.set('invert', config.invert)
    config_env.set('render_save', config.render_save)

    config_env.set('mode', universe.AgentMode.replay)
    config_env.set('case_ids', [os.path.join(dataset_dir, f'{i}.txt') for i in range(num_cases)])
    config_env.set('dataset_cls', DatasetReplay)
    config_env.set('recorder_cls', universe.PseudoRecorder)

    model_name = 'PseudoMethod' + '-' + Env.__name__
    writer = rllib.basic.create_dir(config, model_name, mode='train')

    config_env.set('path_pack', config.path_pack)
    config_env.set('dataset_name', config.dataset_name)

    env = Env(config.env, writer, env_index=0)


    # plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal', adjustable='box')
    

    fig_3d = plt.figure(figsize=(10,10), dpi=100)
    ax_3d = fig_3d.add_subplot(1,1,1, projection='3d')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('time')



    for i in range(num_cases):
        traj = run_one_episode(env, ax, ax_3d)[:-1]
        x = [s.x for s in traj]
        y = [s.y for s in traj]
        ax.plot(x, y, '-')

        z = np.arange(len(traj)) /100 *z_lim + z_start
        ax_3d.plot3D(x, y, z, '-')
        plt.pause(0.001)

    # plt.pause(0.001)
    plt.show(block=False)
    import pdb; pdb.set_trace()

    fig.savefig(os.path.join(env.output_dir, f'{1}.png'))
    fig.savefig(os.path.join(env.output_dir, f'{1}.pdf'))
    fig_3d.savefig(os.path.join(env.output_dir, f'{2}.png'))
    fig_3d.savefig(os.path.join(env.output_dir, f'{2}.pdf'))
