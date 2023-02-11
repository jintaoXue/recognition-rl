import rllib

import numpy as np
import os
import copy


import universe
from universe.carla.dataset import DatasetInteractive as dataset_cls
from utils import scenarios_template
from utils import scenarios_merge


from config.bottleneck_evaluate import config_env_evaluate

from config import merge



####################################################################################
### multi agent ####################################################################
####################################################################################

config_env__with_character = copy.copy(merge.config_env__with_character)
config_env__with_character.update(config_env_evaluate)
config_env__with_character.scenario_cls = scenarios_merge.ScenarioMergeEvaluate


config_env__with_character_assign = copy.copy(config_env__with_character)
config_env__with_character_assign.scenario_cls = scenarios_merge.ScenarioMergeEvaluate_assign


config_env__fix_svo = copy.copy(merge.config_env__with_character)
config_env__fix_svo.update(config_env_evaluate)
config_env__fix_svo.scenario_cls = scenarios_merge.ScenarioBottleneckEvaluate_fix_our_others

####################################################################################
### single agent ###################################################################
####################################################################################


config_env__neural_background = copy.copy(merge.config_env__neural_background)
config_env__neural_background.update(config_env_evaluate)
config_env__neural_background.scenario_cls = scenarios_merge.ScenarioMergeEvaluate_without_mismatch


config_env__idm_background = copy.copy(merge.config_env)
config_env__idm_background.update(config_env_evaluate)
config_env__idm_background.scenario_cls = scenarios_merge.ScenarioMergeEvaluate


