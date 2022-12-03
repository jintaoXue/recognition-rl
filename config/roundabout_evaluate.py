import rllib

import numpy as np
import os
import copy


import universe
from universe.carla.dataset import DatasetInteractive as dataset_cls
from utils import scenarios_template
from utils import scenarios_roundabout


from config.bottleneck_evaluate import config_env_evaluate

from config import roundabout



####################################################################################
### multi agent ####################################################################
####################################################################################



config_env__with_character = copy.copy(roundabout.config_env__with_character)
config_env__with_character.update(config_env_evaluate)
config_env__with_character.scenario_cls = scenarios_roundabout.ScenarioRoundaboutEvaluate





config_env__with_character_assign = copy.copy(config_env__with_character)
config_env__with_character_assign.scenario_cls = scenarios_roundabout.ScenarioRoundaboutEvaluate_assign




####################################################################################
### single agent ###################################################################
####################################################################################


config_env__neural_background = copy.copy(roundabout.config_env__neural_background)
config_env__neural_background.update(config_env_evaluate)
config_env__neural_background.scenario_cls = scenarios_roundabout.ScenarioRoundaboutEvaluate_without_mismatch


config_env__idm_background = copy.copy(roundabout.config_env)
config_env__idm_background.update(config_env_evaluate)
config_env__idm_background.scenario_cls = scenarios_roundabout.ScenarioRoundaboutEvaluate




