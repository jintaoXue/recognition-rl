import rllib

import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import seaborn as sns
import pandas as pd



# https://zhuanlan.zhihu.com/p/158751106


### ISAC, adaptive, multi-scenario
many_characters_data = [
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-13:24:02----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-13:54:10----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508000-evaluate/',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-14:16:44----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-14:45:51----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508000-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-13:24:14----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-13:52:13----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-14:13:21----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-14:37:25----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508200-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-13:24:28----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-13:56:39----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-14:18:20----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-14:46:17----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508400-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-13:24:37----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-13:56:56----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-14:19:22----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-14:47:51----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508600-evaluate',



    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-15:25:52----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-15:56:18----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-16:17:03----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-16:41:54----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-508800-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-15:26:07----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-15:58:00----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-16:19:42----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-16:46:54----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509000-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-15:26:21----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-15:58:42----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-16:19:42----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-16:46:54----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509200-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-15:26:36----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-15:57:16----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-16:17:48----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-07-16:44:03----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario-509400-evaluate',
]



### ISAC, adaptive, bottleneck
many_characters_data = [
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-20:24:25----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-595800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-20:24:21----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-596000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-20:24:07----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-596200-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-20:24:01----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-596400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-19:49:20----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-596600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-19:49:06----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-596800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-19:49:04----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-597000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-19:48:50----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-597200-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-06-20:59:00----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-601000-evaluate',



    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-10:43:36----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-945800-evaluate',


    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-14:56:13----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:23----ray_isac_adaptive_character__bottleneck-1078600-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-15:47:12----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-1286000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-15:47:26----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-1285800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-15:47:30----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-1285600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-15:47:42----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck-1286200-evaluate',

]

### ISAC, adaptive, intersection
many_characters_data = [
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-08-23:36:13----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-07-23:19:44----ray_isac_adaptive_character__intersection-360000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-09-00:10:28----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-07-23:19:44----ray_isac_adaptive_character__intersection-360600-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-09-12:59:04----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-07-23:20:15----ray_isac_adaptive_character__intersection-527600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-09-12:58:53----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-07-23:20:15----ray_isac_adaptive_character__intersection-527800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-09-11:25:45----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-07-23:20:15----ray_isac_adaptive_character__intersection-528000-evaluate',





    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-09-22:52:41----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-07-23:20:15----ray_isac_adaptive_character__intersection-614600-evaluate',

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-00:34:45----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-10-22:40:25----ray_isac_adaptive_character__multi_scenario-423800-evaluate',



    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-16:07:31----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-10-22:42:06----ray_isac_adaptive_character__multi_scenario-706600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-16:42:49----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-10-22:42:06----ray_isac_adaptive_character__multi_scenario-707800-evaluate',

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-11:33:00----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-10-22:42:06----ray_isac_adaptive_character__multi_scenario-713600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-13-12:56:37----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-10-22:42:06----ray_isac_adaptive_character__multi_scenario-713800-evaluate',






]


### ISAC, adaptive, bottleneck
many_characters_data = [
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-00:39:26----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-10-22:42:06----ray_isac_adaptive_character__multi_scenario-719200-evaluate',

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-15:13:53----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-864000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-15:27:00----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-860200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-16:13:37----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-17:18:00----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865800-evaluate',

]

### ISAC, adaptive, roundabout
many_characters_data = [

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-20:25:17----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-20:25:28----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-20:25:32----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866200-evaluate',


    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-20:59:43----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-868000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-20:59:57----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-868200-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-20:59:59----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-868400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:00:09----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-868600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:00:10----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-868800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:00:23----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-869000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:00:24----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-869200-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:00:27----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-869400-evaluate',


]


### ISAC, adaptive, merge
many_characters_data = [
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:59:11----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-698600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:59:24----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-698400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:59:50----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-698200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:00:04----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-698000-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:00:06----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-698800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:00:20----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-699000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:00:27----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-699200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:00:38----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-699400-evaluate',


    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:47:11----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-699600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:47:17----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-699800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:47:33----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-700000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:47:41----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-700200-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:47:50----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-700400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:48:00----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-700600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:48:04----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-700800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-22:48:17----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-701000-evaluate',





    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:40:34----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:40:43----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:40:50----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:41:00----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866000-evaluate',
    
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:41:01----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:41:06----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:41:17----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:41:23----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866800-evaluate',



]


### ISAC, adaptive, intersection
many_characters_data = [
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-14:28:05----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-296200-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-10:43:40----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-344400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-10:44:53----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-344600-evaluate',
    
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-14:28:44----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-401000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-14:28:44----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-401200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-15:53:31----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-402600-evaluate',

    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-16:35:38----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-422400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-16:35:52----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-422600-evaluate',

]





### ISAC, robust, bottleneck
many_characters_data = [
    ### adaptive
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-17:18:00----evaluate_ray_isac_adaptive_character_assign__bottleneck--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-865800-evaluate',



    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-20:07:44----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-568000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-20:07:50----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-568200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-20:08:04----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-567400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-20:08:20----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-567000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-20:08:38----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-566800-evaluate',



    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-21:24:38----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-568400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-21:24:45----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-568600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-21:24:54----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-568800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-21:25:03----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-569000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-21:25:07----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-569200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-21:25:13----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-569400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-21:25:23----evaluate_ray_isac_robust_character_assign__bottleneck--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-569600-evaluate',





]

### ISAC, robust, intersection
many_characters_data = [
    ### adaptive
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-16:35:52----evaluate_ray_isac_adaptive_character_assign__intersection--from--2022-09-14-10:25:41----ray_isac_adaptive_character__intersection-422600-evaluate',



    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:18:02----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-593600-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:18:15----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-593400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:18:20----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-593200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:18:31----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-593000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:18:36----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-593800-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:18:51----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-594000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:19:05----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-594200-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-22:19:10----evaluate_ray_isac_robust_character_assign__intersection--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-594400-evaluate',


]


### ISAC, robust, merge
many_characters_data = [
    ### adaptive
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-23:41:01----evaluate_ray_isac_adaptive_character_assign__merge--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-866200-evaluate',



    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:13:36----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-541000-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:13:48----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-541200-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:13:50----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-541400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:14:11----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-541600-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:14:14----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-541800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:14:27----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-540800-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:14:39----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-540600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:14:42----evaluate_ray_isac_robust_character_assign__merge--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-540400-evaluate',


]


### ISAC, robust, roundabout
many_characters_data = [
    ### adaptive
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-14-21:00:27----evaluate_ray_isac_adaptive_character_assign__roundabout--from--2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2-869400-evaluate',



    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:59:14----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-580000-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:59:27----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-580200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:59:36----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-580400-evaluate',
    '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:59:52----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-579800-evaluate',

    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-15-23:59:57----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-579600-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-00:00:09----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-579400-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-00:00:11----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-579200-evaluate',
    # '~/Downloads/log/download_log/IndependentSAC_v0-EnvInteractiveMultiAgent-Evaluate/2022-09-16-00:00:26----evaluate_ray_isac_robust_character_assign__roundabout--from--2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario-579000-evaluate',



]





def assign_character():
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

    fig.savefig(os.path.join('results', 'character_new.png'))
    fig.savefig(os.path.join('results', 'character_new.pdf'))


    return











if __name__ == "__main__":
    assign_character()

