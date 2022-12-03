import rllib

import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import seaborn as sns
import pandas as pd



# https://zhuanlan.zhihu.com/p/158751106



many_data = [
    ### adaptive
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-11:04:30----evaluate_ray_isac_adaptive_character__bottleneck--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-12:30:33----evaluate_ray_isac_adaptive_character__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-422600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-11:04:44----evaluate_ray_isac_adaptive_character__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866200-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-11:05:28----evaluate_ray_isac_adaptive_character__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-869400-evaluate',

    'None',
    'None',

    ### robust
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-11:02:23----evaluate_ray_isac_robust_character__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-568800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-12:30:37----evaluate_ray_isac_robust_character__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-593800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-11:03:10----evaluate_ray_isac_robust_character__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-541000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-11:03:30----evaluate_ray_isac_robust_character__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-579800-evaluate',

    'None',
    'None',
    'None',
    'None',



    ### robust copo
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-11:19:03----evaluate_ray_isac_robust_character_copo__bottleneck--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-11:19:23----evaluate_ray_isac_robust_character_copo__intersection--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-11:19:40----evaluate_ray_isac_robust_character_copo__merge--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-11:19:51----evaluate_ray_isac_robust_character_copo__roundabout--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',

    'None',

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-12:46:32----evaluate_ray_isac_robust_character_copo__bottleneck--manual--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-12:46:36----evaluate_ray_isac_robust_character_copo__intersection--manual--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-12:46:47----evaluate_ray_isac_robust_character_copo__merge--manual--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-12:46:50----evaluate_ray_isac_robust_character_copo__roundabout--manual--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    'None',

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-13:10:31----evaluate_ray_isac_robust_character_copo__bottleneck--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-152600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-13:10:33----evaluate_ray_isac_robust_character_copo__intersection--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-152600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-13:10:34----evaluate_ray_isac_robust_character_copo__merge--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-152600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-13:10:37----evaluate_ray_isac_robust_character_copo__roundabout--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-152600-evaluate',

    'None',

    ### final
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-14:04:54----evaluate_ray_isac_robust_character_copo__bottleneck--manual--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-14:04:55----evaluate_ray_isac_robust_character_copo__intersection--manual--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-14:04:57----evaluate_ray_isac_robust_character_copo__merge--manual--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-14:05:00----evaluate_ray_isac_robust_character_copo__roundabout--manual--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',

    'None',
    'None',



    ### copo adv

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-18-14:17:45----evaluate_ray_isac_robust_character_copo_adv__bottleneck--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-18-14:18:18----evaluate_ray_isac_robust_character_copo_adv__intersection--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-18-14:18:38----evaluate_ray_isac_robust_character_copo_adv__merge--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-18-14:19:06----evaluate_ray_isac_robust_character_copo_adv__roundabout--from--2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario-126000-evaluate',


    'None',
    'None',



    ### no character

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-15:12:21----evaluate_ray_isac_no_character__bottleneck--from--2022-09-13-22:44:21----ray_isac_no_character__multi_scenario-696400-evaluate',

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-14:42:09----evaluate_ray_isac_no_character__intersection--from--2022-09-13-22:44:21----ray_isac_no_character__multi_scenario-696800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-14:42:39----evaluate_ray_isac_no_character__merge--from--2022-09-13-22:44:21----ray_isac_no_character__multi_scenario-696800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-14:42:53----evaluate_ray_isac_no_character__roundabout--from--2022-09-13-22:44:21----ray_isac_no_character__multi_scenario-696800-evaluate',


    'None',
    'None',

    ### idm
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-15:39:19----evaluate_ray_isac_idm__bottleneck-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-15:39:32----evaluate_ray_isac_idm__intersection-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-15:39:47----evaluate_ray_isac_idm__merge-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-15:39:59----evaluate_ray_isac_idm__roundabout-evaluate',


]





def assign_character():
    keys = ['success', 'collision', 'off_road', 'off_route', 'wrong_lane', 'speed']
    titles = ['Success', 'Collision', 'Off Road', 'Off Route', 'Wrong Lane', 'Speed']

    keys = ['success', 'collision', 'off_road', 'off_route', 'wrong_lane']
    titles = ['Success', 'Collision', 'Off Road', 'Off Route', 'Wrong Lane']


    def calculate_average(data_path, clip=None):
        import numpy as np

        data = np.loadtxt(data_path, delimiter=' ')

        if clip:
            data = data[:clip]

        return np.average(data[:,1])




    for data_dir in many_data:
        if data_dir == 'None':
            print('')
            continue
            
        label = data_dir.split('----evaluate_')[-1]
        data = {}
        for key, title in zip(keys, titles):
            data[key] = {'title': title, 'data': []}
        for i in range(5):
            for key, title in zip(keys, titles):
                
                _data_path = os.path.expanduser(os.path.join(data_dir, f'log/env{i}/*log/env{i}*/*{key}*'))
                # print(_data_path)
                data_path = glob.glob(_data_path)[0]
                data[key]['data'].append(calculate_average(data_path))


        data_str = f'& {label} '
        for key, _data in data.items():
            d = _data['data']
            mean = np.mean(d) *100
            std = np.std(d) *100

            data_str += ' & {} $\pm$ {} \% '.format(np.around(mean, 1), np.around(std, 1))
        
        print(data_str)



    return











if __name__ == "__main__":
    assign_character()

