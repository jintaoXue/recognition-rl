import rllib

import numpy as np
import os
import copy


import universe
from universe.carla.dataset import DatasetInteractive as dataset_cls
from utils import scenarios_template
from utils import scenarios_bottleneck
from utils import perception_downsample
from utils import agents_master
from config import bottleneck




config_env_evaluate = rllib.basic.YamlConfig(
    num_vehicles_range=rllib.basic.BaseData(min=20, max=20),
    # recorder_cls=universe.Recorder,
)


####################################################################################
### multi agent ####################################################################
####################################################################################


config_env__with_character = copy.copy(bottleneck.config_env__with_character)
config_env__with_character.update(config_env_evaluate)
config_env__with_character.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate




config_env__with_character_assign = copy.copy(config_env__with_character)
config_env__with_character_assign.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate_assign




####################################################################################
### single agent ###################################################################
####################################################################################

config_env__neural_background = copy.copy(bottleneck.config_env__neural_background)
config_env__neural_background.update(config_env_evaluate)
config_env__neural_background.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate_without_mismatch


config_env__idm_background = copy.copy(bottleneck.config_env)
config_env__idm_background.update(config_env_evaluate)
config_env__idm_background.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate



config_env__neural_background_fix = copy.copy(bottleneck.config_env__neural_background)
config_env__neural_background_fix.update(config_env_evaluate)
config_env__neural_background_fix.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate_fix_our_others


config_env__idm_background_fix = copy.copy(bottleneck.config_env)
config_env__idm_background_fix.update(config_env_evaluate)
config_env__idm_background_fix.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate_fix_our_others

config_env__neural_background_sampling = copy.copy(bottleneck.config_env__neural_background_sampling)
config_env__neural_background_sampling.update(config_env_evaluate)
config_env__neural_background_sampling.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate_without_mismatch



config_env__neural_background_same_other_svo = copy.copy(bottleneck.config_env__neural_background_same_other_svo)
config_env__neural_background_same_other_svo.update(config_env_evaluate)
config_env__neural_background_same_other_svo.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluateFixOtherSvo


#muli agent model in EnV
config_env__new_action_same_other = copy.copy(bottleneck.config_env__new_action_background)
config_env__new_action_same_other.update(config_env_evaluate)
config_env__new_action_same_other.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluateFixOtherSvo

#ILAttn
config_env__neural_background_mix = copy.copy(bottleneck.config_env__neural_background_mix)
config_env__neural_background_mix.update(config_env_evaluate)
config_env__neural_background_mix.scenario_cls = scenarios_bottleneck.ScenarioBottleneckDiverseEvaluate

#RILEnvM
config_env__multiact__mixbkgrd = copy.copy(bottleneck.config_env__multiact__mixbkgrd)
config_env__multiact__mixbkgrd.update(config_env_evaluate)
config_env__multiact__mixbkgrd.scenario_cls = scenarios_bottleneck.ScenarioBottleneckDiverseEvaluate
# config_env__multiact__mixbkgrd.rule_vehicle_cls = agents_master.EndToEndVehicleWithCharacterBackground

config_env__new_action_multi_svo_same_other_svo = copy.copy(bottleneck.config_env__new_action_multi_svo)
config_env__new_action_multi_svo_same_other_svo.update(config_env_evaluate)
config_env__new_action_multi_svo_same_other_svo.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluateFixOtherSvo
# config_env__new_action_multi_svo_same_other_svo.scenario_randomization_cls = scenarios_template.ScenarioRandomizationDivese


config_env__new_action_fix_svo = copy.copy(bottleneck.config_env__new_action_background)
config_env__new_action_fix_svo.update(config_env_evaluate)
config_env__new_action_fix_svo.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate_fix_our_others

config_env__new_actions_fix_svo = copy.copy(bottleneck.config_env__new_action_multi_svo)
config_env__new_actions_fix_svo.update(config_env_evaluate)
config_env__new_actions_fix_svo.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate_fix_our_others

config_env__supervise_fix_svo = copy.copy(bottleneck.config_env__neural_background_same_other_svo)
config_env__supervise_fix_svo.update(config_env_evaluate)
config_env__supervise_fix_svo.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate_fix_our_others