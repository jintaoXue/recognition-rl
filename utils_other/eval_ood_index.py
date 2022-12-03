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
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:18:38----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-755400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:18:45----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-755600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:18:48----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-755800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:18:58----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-756000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:19:01----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-756200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:19:07----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-756400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:19:15----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-756600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:19:21----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-756800-evaluate',


    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:34:36----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-512800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:34:46----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-513000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:34:52----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-513200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:34:57----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-513400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:35:02----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-513600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:35:10----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-513800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:35:14----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-514000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:35:28----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-512600-evaluate',


    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:10----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-757000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:20----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-757200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:21----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-757400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:34----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-757600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:39----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-757800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:46----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-758000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:19:51----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-758200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-17:20:03----evaluate__full_stack_background__intersection--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-758400-evaluate',




    'None',
    'None',
    'None',


    ### no character
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-04:06:13----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-393200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-04:06:21----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-393000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-04:06:38----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-392800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-04:06:43----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-392600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-04:07:01----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-392400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-04:07:20----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-392200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-04:07:44----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-393400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-04:07:57----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-393600-evaluate',

    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:11:37----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-735800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:11:48----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-735600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:11:59----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-735400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:12:19----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-735200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:12:32----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-735000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:12:44----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-734800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:12:56----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-734600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:13:10----evaluate__full_stack_background__intersection--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-734400-evaluate',


    'None',
    'None',
    'None',


    ### robust copo
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:07:35----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-412200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:07:57----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-412000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:07:59----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-411800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:01----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-411600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:05----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-411400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:10----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-411000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:10----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-411200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:19----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-410800-evaluate',


    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:23:49----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-584800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:24:18----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-584600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:24:28----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-584400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:24:39----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-584200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:24:53----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-585000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:25:11----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-585200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:25:18----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-585400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-13:25:25----evaluate__full_stack_background__intersection--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-585600-evaluate',



    'None',
    'None',
    'None',


    ### adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:09:41----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1114000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:09:51----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1114200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:10:14----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1114400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:10:19----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1114600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:10:35----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1113800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:10:46----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1113600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:15:12----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-696600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:15:15----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-696800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:15:27----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:15:29----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:15:40----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:15:43----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:15:55----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-11:16:02----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-698000-evaluate',



    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:43:29----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-698200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:43:29----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-698400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:43:31----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-698600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:43:50----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-698800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:44:02----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-699000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:44:04----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-699200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:44:14----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-699400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-14:44:19----evaluate__full_stack_background__intersection--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-699600-evaluate',



    'None',
    'None',
    'None',


    ### duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:47----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:51----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:55----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:08:58----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:09:04----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-424600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:09:04----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-424800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:09:09----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-424400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-05:09:14----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-424200-evaluate',

    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-10:28:32----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-19-10:28:50----evaluate__full_stack_background__intersection--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-319400-evaluate',





]




### SAC, merge
many_data__merge = [

    ### idm
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:02:54----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:03:03----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:03:10----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-14:03:16----evaluate__full_stack_background__merge--idm--from--2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario-700000-evaluate',


    'None',
    'None',
    'None',

    ### no character
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:27:28----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-610000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:27:35----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-610200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:27:46----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-609800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:27:55----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-609600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:28:04----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-609400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:28:09----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-609200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:28:24----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-610400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:28:29----evaluate__full_stack_background__merge--no_character--from--2022-09-16-21:18:51----ray_sac__no_character__multi_scenario-610600-evaluate',

    'None',
    'None',
    'None',


    ### robust copo
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:30:14----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-631200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:30:25----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-631400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:30:39----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-631600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:30:59----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-631000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:11----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:22----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:31----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-15:31:43----evaluate__full_stack_background__merge--robust_copo--from--2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario-630200-evaluate',


    'None',
    'None',
    'None',




    ### adaptive
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-20:22:14----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1016000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-20:22:54----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1016200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-20:23:03----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1016400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-20:23:15----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1015800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-20:23:26----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1015600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-20:23:36----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1015400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-20:23:45----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1015200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-20:23:56----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-1015000-evaluate',

    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:24:32----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-696800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:24:37----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:24:47----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:24:50----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:25:02----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-696600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:25:10----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-696400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:25:21----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-696200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-21:25:38----evaluate__full_stack_background__merge--adaptive--from--2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario-697600-evaluate',



    'None',
    'None',
    'None',


    ### duplicity
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:52:49----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:52:59----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:53:09----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:53:19----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-425000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:53:33----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-424800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:53:44----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-424600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:53:56----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-424400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-13:54:07----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-424200-evaluate',


    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-16:57:25----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-487000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-16:57:31----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-487200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-16:57:42----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-487400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-16:57:54----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-487600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-16:58:03----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-487800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-16:58:14----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-488000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-16:58:19----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-486800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-16:58:32----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-486600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:52:39----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-580000-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:52:49----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-580200-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:52:59----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-580400-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:53:09----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-580600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:53:20----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-580800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:54:04----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-579800-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:54:15----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-579600-evaluate',
    '~/Downloads/log/download_log/SAC-EnvInteractiveSingleAgent-Evaluate/2022-09-18-17:54:28----evaluate__full_stack_background__merge--duplicity--from--2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background-579400-evaluate',




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
                data_str += ' & {} \% '.format(np.around(mean, 1))
        
            print(data_str)
        print('')



    return











if __name__ == "__main__":
    assign_character()

