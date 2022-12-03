import rllib

import numpy as np
import os
import copy


import universe
from universe.carla.dataset import DatasetInteractive as dataset_cls
from utils import scenarios_template
from utils import scenarios_intersection


from config.bottleneck_evaluate import config_env_evaluate

from config import intersection



####################################################################################
### multi agent ####################################################################
####################################################################################



config_env__with_character = copy.copy(intersection.config_env__with_character)
config_env__with_character.update(config_env_evaluate)
config_env__with_character.scenario_cls = scenarios_intersection.ScenarioIntersectionEvaluate





config_env__with_character_assign = copy.copy(config_env__with_character)
config_env__with_character_assign.scenario_cls = scenarios_intersection.ScenarioIntersectionEvaluate_assign




####################################################################################
### single agent ###################################################################
####################################################################################

config_env__neural_background = copy.copy(intersection.config_env__neural_background)
config_env__neural_background.update(config_env_evaluate)
config_env__neural_background.scenario_cls = scenarios_intersection.ScenarioIntersectionEvaluate_without_mismatch


config_env__idm_background = copy.copy(intersection.config_env)
config_env__idm_background.update(config_env_evaluate)
config_env__idm_background.scenario_cls = scenarios_intersection.ScenarioIntersectionEvaluate



