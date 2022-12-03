import rllib

import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import seaborn as sns
import pandas as pd



# https://zhuanlan.zhihu.com/p/158751106





many_characters_data_bottleneck = [
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-17:18:00----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-21:24:54----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-568800-evaluate',


]

many_characters_data_intersection = [
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-16:35:38----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-422400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-16:35:52----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-422600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:18:36----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-593800-evaluate',


]

many_characters_data_merge = [
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:41:01----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866200-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:13:36----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-541000-evaluate',

]

many_characters_data_roundabout = [
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:00:27----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-869400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:59:52----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-579800-evaluate',

]



### ISAC, adaptive
many_characters_data = [
    ### bottleneck
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-17:18:00----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865800-evaluate',

    ### intersection
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-16:35:38----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-422400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-16:35:52----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-422600-evaluate',

    ### merge
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:41:01----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866200-evaluate',

    ### roundabout
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:00:27----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-869400-evaluate',

]


### ISAC, robust
many_characters_data = [
    ### bottleneck
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-21:24:54----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-568800-evaluate',

    ### intersection
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:18:36----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-593800-evaluate',

    ### merge
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:13:36----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-541000-evaluate',

    ### roundabout
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:59:52----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-579800-evaluate',

]






def assign_character(many_characters_data, scenario):
    keys = ['success', 'collision', 'off_road', 'off_route', 'wrong_lane', 'speed']
    titles = ['Success', 'Collision', 'Off Road', 'Off Route', 'Wrong Lane', 'Speed']
    characters = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


    def calculate_average(data_path, clip=None):
        import numpy as np

        data = np.loadtxt(data_path, delimiter=' ')

        if clip:
            data = data[:clip]

        return np.average(data[:,1])



    df = pd.DataFrame()

    for characters_data in many_characters_data:
        label = characters_data.split('assign__')[-1]
        for i, character in enumerate(characters):
            for key, title in zip(keys, titles):
                
                _data_path = os.path.expanduser(os.path.join(characters_data, f'log/env{i}/*log/env{i}*/*{key}*'))
                print(_data_path)
                data_path = glob.glob(_data_path)[0]
                data = [calculate_average(data_path)]

                for d in data:
                    df = df.append(pd.Series(rllib.basic.Data(character=character, index=title, value=d, label=label).to_dict()), ignore_index=True)




    # sns.set()
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(15,8+2*len(keys)), dpi=100)

    fig.clear()
    axes = fig.subplots(len(keys), 1)




    for ax, key, title in zip(axes, keys, titles):
        data = df[df['index'] == title]

        color = 'cornflowerblue'
        sns.lineplot(ax=ax, data=data, x='character', y='value', hue='label', color=color, palette="muted", label='')
        ax.set_ylabel(title)
        ax.get_legend().remove()

    axes[-1].set_xlabel('character')
    axes[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5))
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0))

    fig.suptitle('index')
    # fig.legend(loc='lower center')
    # fig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig.savefig(os.path.join('results', f'svo_isac_{scenario}.png'))
    fig.savefig(os.path.join('results', f'svo_isac_{scenario}.pdf'))


    return











if __name__ == "__main__":
    assign_character(many_characters_data_bottleneck, 'bottleneck')
    assign_character(many_characters_data_intersection, 'intersection')
    assign_character(many_characters_data_merge, 'merge')
    assign_character(many_characters_data_roundabout, 'roundabout')

