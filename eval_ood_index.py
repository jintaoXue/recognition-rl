import rllib

import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import seaborn as sns
import pandas as pd



# https://zhuanlan.zhihu.com/p/158751106



### SAC, bottleneck
many_data = [
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:08:23----evaluate__full_stack_background__bottleneck--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-735800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:09:21----evaluate__full_stack_background__bottleneck--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-735600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:09:29----evaluate__full_stack_background__bottleneck--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-735400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:09:36----evaluate__full_stack_background__bottleneck--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-735200-evaluate',

    'None',

    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:15:44----evaluate__full_stack_background__bottleneck--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-318800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:16:06----evaluate__full_stack_background__bottleneck--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:16:15----evaluate__full_stack_background__bottleneck--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:16:24----evaluate__full_stack_background__bottleneck--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:16:59----evaluate__full_stack_background__bottleneck--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-318600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:17:10----evaluate__full_stack_background__bottleneck--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-318400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:17:40----evaluate__full_stack_background__bottleneck--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-318200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-17-14:17:56----evaluate__full_stack_background__bottleneck--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-318000-evaluate',


    'None',
    'None',



]


### SAC, intersection
many_data = [
    ### idm
    ### 1. idm
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:34:57----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-513400-evaluate',
    ### 2. no_character
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:20----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-757200-evaluate',
    ### 3. robust_copo
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:20----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-757200-evaluate',
    ### 4. explicit_adv
    
    ### 5. adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:34----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-757600-evaluate',
    ### 6. duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:34----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-757600-evaluate',



    'None',
    'None',
    'None',



    ### no character
    ### 1. idm
    
    ### 2. no_character
    
    ### 3. robust_copo
    
    ### 4. explicit_adv
    
    ### 5. adaptive
    
    ### 6. duplicity


    'None',
    'None',
    'None',


    ### robust copo
    ### 1. idm
    
    ### 2. no_character
    
    ### 3. robust_copo
    
    ### 4. explicit_adv
    
    ### 5. adaptive
    
    ### 6. duplicity


    'None',
    'None',
    'None',


    ### adaptive
    ### 1. idm
    
    ### 2. no_character
    
    ### 3. robust_copo
    
    ### 4. explicit_adv
    
    ### 5. adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:44:14----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-699400-evaluate',
    ### 6. duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:44:14----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-699400-evaluate',



    'None',
    'None',
    'None',

    ### duplicity
    ### 1. idm
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:55----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425200-evaluate',
    ### 2. no_character
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-10:28:32----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319200-evaluate',
    ### 3. robust_copo
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-10:28:32----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319200-evaluate',
    ### 4. explicit_adv
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-10:28:32----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319200-evaluate',
    ### 5. adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-10:28:32----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319200-evaluate',
    ### 6. duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-10:28:32----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319200-evaluate',



]




### SAC, merge
many_data__merge = [

    ### idm
    ### 1. idm
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:03:10----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700200-evaluate',
    ### 2. no_character
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:03:10----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700200-evaluate',
    ### 3. robust_copo
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:03:10----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700200-evaluate',
    ### 4. explicit_adv
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:03:10----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700200-evaluate',
    ### 5. adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:03:10----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700200-evaluate',
    ### 6. duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:03:10----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700200-evaluate',


    'None',
    'None',
    'None',

    ### no character
    ### 1. idm
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:27:46----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-609800-evaluate',
    ### 2. no_character
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:27:35----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-610200-evaluate',
    ### 3. robust_copo
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:28:09----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-609200-evaluate',
    ### 4. explicit_adv
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:27:46----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-609800-evaluate',
    ### 5. adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:27:55----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-609600-evaluate',
    ### 6. duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:27:55----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-609600-evaluate',

    'None',
    'None',
    'None',


    ### robust copo
    ### 1. idm
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:43----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630200-evaluate',
    ### 2. no_character
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:43----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630200-evaluate',
    ### 3. robust_copo
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:43----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630200-evaluate',
    ### 4. explicit_adv
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:31----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630400-evaluate',
    ### 5. adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:43----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630200-evaluate',
    ### 6. duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:43----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630200-evaluate',



    'None',
    'None',
    'None',


    ### adaptive
    ### 1. idm
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-20:22:14----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1016000-evaluate',
    ### 2. no_character
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:25:02----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-696600-evaluate',
    ### 3. robust_copo
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:24:47----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697200-evaluate',
    ### 4. explicit_adv
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:24:37----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697000-evaluate',
    ### 5. adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:24:37----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697000-evaluate',
    ### 6. duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:24:37----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697000-evaluate',


    'None',
    'None',
    'None',



    ### duplicity
    ### 1. idm
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:54:04----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-579800-evaluate',
    ### 2. no_character
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:53:20----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-580800-evaluate',
    ### 3. robust_copo
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:53:20----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-580800-evaluate',
    ### 4. explicit_adv
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:53:44----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-424600-evaluate',
    ### 5. adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:53:20----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-580800-evaluate',
    ### 6. duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:52:59----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425400-evaluate',



]






def assign_character():
    keys = ['success', 'collision', 'off_road', 'off_route', 'wrong_lane', 'speed']
    titles = ['Success', 'Collision', 'Off Road', 'Off Route', 'Wrong Lane', 'Speed']


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
        
        num_envs = len(glob.glob(os.path.expanduser(os.path.join(data_dir, 'log/env*/'))))
        for i in range(num_envs):
            env_name = glob.glob(os.path.expanduser(os.path.join(data_dir, f'log/env{i}/*log/env{i}*/')))[0].split('/')[-2]
            label_env = label + '---' + env_name

            data = {}
            for key, title in zip(keys, titles):
                data[key] = {'title': title, 'data': []}

            for key, title in zip(keys, titles):
                
                _data_path = os.path.expanduser(os.path.join(data_dir, f'log/env{i}/*log/env{i}*/*{key}*'))
                # print(_data_path)
                data_path = glob.glob(_data_path)[0]
                data[key]['data'].append(calculate_average(data_path))


            data_str = f'& {label_env} '
            for key, _data in data.items():
                d = _data['data']
                mean = np.mean(d) *100
                std = np.std(d) *100

                # data_str += ' & {} $\pm$ {} \% '.format(np.around(mean, 1), np.around(std, 1))
                data_str += ' & {}\% '.format(np.around(mean, 1))
        
            print(data_str)
        print('')



    return











if __name__ == "__main__":
    assign_character()

