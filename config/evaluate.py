import rllib
import universe

import copy

from utils import scenarios_template
from utils import scenarios_bottleneck, scenarios_intersection, scenarios_merge, scenarios_roundabout
from . import train


config_env = rllib.basic.YamlConfig(
    num_vehicles_range=rllib.basic.BaseData(min=20, max=20),
    # recorder_cls=universe.Recorder,
)




############################################################################
##### bottleneck ###########################################################
############################################################################


config_bottleneck__with_character = copy.copy(train.config_bottleneck__with_character)
config_bottleneck__with_character.update(config_env)
config_bottleneck__with_character.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate

# config_bottleneck__with_character_share = copy.copy(config_bottleneck__with_character)
# config_bottleneck__with_character_share.scenario_randomization_cls = scenarios_template.ScenarioRandomization_share_character


config_bottleneck__with_character_assign = copy.copy(config_bottleneck__with_character)
config_bottleneck__with_character_assign.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate_assign



### single agent

config_bottleneck__neural_background = copy.copy(train.config_bottleneck__neural_background)
config_bottleneck__neural_background.update(config_env)

config_bottleneck__idm_background = copy.copy(train.config_bottleneck)
config_bottleneck__idm_background.update(config_env)
config_bottleneck__idm_background.scenario_cls = scenarios_bottleneck.ScenarioBottleneckEvaluate




############################################################################
##### intersection #########################################################
############################################################################



config_intersection__with_character = copy.copy(train.config_intersection__with_character)
config_intersection__with_character.update(config_env)
config_intersection__with_character.scenario_cls = scenarios_intersection.ScenarioIntersectionEvaluate



config_intersection__with_character_assign = copy.copy(config_intersection__with_character)
config_intersection__with_character_assign.scenario_cls = scenarios_intersection.ScenarioIntersectionEvaluate_assign




############################################################################
##### merge ################################################################
############################################################################




config_merge__with_character = copy.copy(train.config_merge__with_character)
config_merge__with_character.update(config_env)
config_merge__with_character.scenario_cls = scenarios_merge.ScenarioMergeEvaluate



config_merge__with_character_assign = copy.copy(config_merge__with_character)
config_merge__with_character_assign.scenario_cls = scenarios_merge.ScenarioMergeEvaluate_assign






############################################################################
##### roundabout ###########################################################
############################################################################



config_roundabout__with_character = copy.copy(train.config_roundabout__with_character)
config_roundabout__with_character.update(config_env)
config_roundabout__with_character.scenario_cls = scenarios_roundabout.ScenarioRoundaboutEvaluate



config_roundabout__with_character_assign = copy.copy(config_roundabout__with_character)
config_roundabout__with_character_assign.scenario_cls = scenarios_roundabout.ScenarioRoundaboutEvaluate_assign




