import rllib

import numpy as np
import os
import copy


import universe
from universe.carla.dataset import DatasetInteractive as dataset_cls
from utils import scenarios_template
from utils import scenarios_bottleneck, scenarios_intersection, scenarios_merge, scenarios_roundabout
from utils import agents_master
from utils import perception
from utils import reward


config_vehicle = rllib.basic.YamlConfig(
    min_velocity=0.0,
    max_velocity=6.0,
    max_acceleration=5.0,
    min_acceleration=-5.0,
    max_steer=np.deg2rad(45),
    wheelbase=2.6,

    # max_throttle=1.0,
    # max_brake=1.0,
    bbx_x=2.1, bbx_y=1.0,
)



############################################################################
##### bottleneck ###########################################################
############################################################################

config_bottleneck = rllib.basic.YamlConfig(
    mode=universe.AgentMode.interactive,
    case_ids=[None],
    map_name='synthetic_v1',
    scenario_name='bottleneck',
    dataset_dir=rllib.basic.BaseData(
        map=os.path.expanduser('~/dataset/carla/maps'),
        path=os.path.expanduser('~/dataset/carla/synthetic_v1-bottleneck/global_path'),
    ),
    dataset_cls=dataset_cls,

    scenario_randomization_cls=scenarios_template.ScenarioRandomization,
    scenario_cls=scenarios_bottleneck.ScenarioBottleneck,
    boundary=rllib.basic.BaseData(x_min=-150, y_min=180, x_max=150, y_max=220),
    center=rllib.basic.BaseData(x=0, y=200),
    num_vehicles_range=rllib.basic.BaseData(min=8, max=20),
    # num_vehicles_range=rllib.basic.BaseData(min=1, max=1),  ### ! warning
    # num_vehicles_range=rllib.basic.BaseData(min=2, max=2),  ### ! warning

    decision_frequency=4, control_frequency=40,
    perception_range=50.0,

    bbx_x=config_vehicle.bbx_x, bbx_y=config_vehicle.bbx_y,
    config_vehicle=config_vehicle,

    agents_master_cls=agents_master.AgentListMaster,
    neural_vehicle_cls=agents_master.EndToEndVehicleWithCharacter,
    rule_vehicle_cls=agents_master.IdmVehicleWithCharacter,
    perception_cls=perception.PerceptionPointNet,
    reward_func=reward.RewardFunctionNoCharacter,

)


config_bottleneck__with_character = copy.copy(config_bottleneck)
config_bottleneck__with_character.reward_func = reward.RewardFunctionWithCharacter

config_bottleneck__with_character_share = copy.copy(config_bottleneck__with_character)
config_bottleneck__with_character_share.scenario_randomization_cls = scenarios_template.ScenarioRandomization_share_character




### single agent

config_bottleneck__neural_background = copy.copy(config_bottleneck)
config_bottleneck__neural_background.scenario_randomization_cls = scenarios_template.ScenarioRandomizationWithoutMismatch
config_bottleneck__neural_background.agents_master_cls = agents_master.AgentListMasterNeuralBackground
config_bottleneck__neural_background.rule_vehicle_cls = agents_master.EndToEndVehicleWithCharacterBackground









############################################################################
##### intersection #########################################################
############################################################################


config_intersection = rllib.basic.YamlConfig(
    mode=universe.AgentMode.interactive,
    case_ids=[None],
    map_name='synthetic_v1',
    scenario_name='intersection',
    dataset_dir=rllib.basic.BaseData(
        map=os.path.expanduser('~/dataset/carla/maps'),
        path=os.path.expanduser('~/dataset/carla/synthetic_v1-intersection/global_path'),
    ),
    dataset_cls=dataset_cls,

    scenario_randomization_cls=scenarios_template.ScenarioRandomization,
    scenario_cls=scenarios_intersection.ScenarioIntersection,
    boundary=rllib.basic.BaseData(x_min=-74, y_min=-74, x_max=74, y_max=74),
    center=rllib.basic.BaseData(x=0, y=0),
    num_vehicles_range=rllib.basic.BaseData(min=8, max=20),
    # num_vehicles_range=rllib.basic.BaseData(min=1, max=1),  ### ! warning
    # num_vehicles_range=rllib.basic.BaseData(min=2, max=2),  ### ! warning

    decision_frequency=4, control_frequency=40,
    perception_range=50.0,

    bbx_x=config_vehicle.bbx_x, bbx_y=config_vehicle.bbx_y,
    config_vehicle=config_vehicle,

    agents_master_cls=agents_master.AgentListMaster,
    neural_vehicle_cls=agents_master.EndToEndVehicleWithCharacter,
    rule_vehicle_cls=agents_master.IdmVehicleWithCharacter,
    perception_cls=perception.PerceptionPointNet,
    reward_func=reward.RewardFunctionNoCharacter,

)


config_intersection__with_character = copy.copy(config_intersection)
config_intersection__with_character.reward_func = reward.RewardFunctionWithCharacter

config_intersection__with_character_share = copy.copy(config_intersection__with_character)
config_intersection__with_character_share.scenario_randomization_cls = scenarios_template.ScenarioRandomization_share_character





############################################################################
##### merge ################################################################
############################################################################


config_merge = rllib.basic.YamlConfig(
    mode=universe.AgentMode.interactive,
    case_ids=[None],
    map_name='synthetic_v1',
    scenario_name='merge',
    dataset_dir=rllib.basic.BaseData(
        map=os.path.expanduser('~/dataset/carla/maps'),
    ),
    dataset_cls=dataset_cls,

    scenario_randomization_cls=scenarios_template.ScenarioRandomization,
    scenario_cls=scenarios_merge.ScenarioMerge,
    boundary=rllib.basic.BaseData(x_min=120, y_min=-140, x_max=220, y_max=-20),
    center=rllib.basic.BaseData(x=170, y=-80),
    num_vehicles_range=rllib.basic.BaseData(min=8, max=20),
    # num_vehicles_range=rllib.basic.BaseData(min=1, max=1),  ### ! warning
    # num_vehicles_range=rllib.basic.BaseData(min=2, max=2),  ### ! warning

    decision_frequency=4, control_frequency=40,
    perception_range=50.0,

    bbx_x=config_vehicle.bbx_x, bbx_y=config_vehicle.bbx_y,
    config_vehicle=config_vehicle,

    agents_master_cls=agents_master.AgentListMaster,
    neural_vehicle_cls=agents_master.EndToEndVehicleWithCharacter,
    rule_vehicle_cls=agents_master.IdmVehicleWithCharacter,
    perception_cls=perception.PerceptionPointNet,
    reward_func=reward.RewardFunctionNoCharacter,

)


config_merge__with_character = copy.copy(config_merge)
config_merge__with_character.reward_func = reward.RewardFunctionWithCharacter

config_merge__with_character_share = copy.copy(config_merge__with_character)
config_merge__with_character_share.scenario_randomization_cls = scenarios_template.ScenarioRandomization_share_character






############################################################################
##### roundabout ###########################################################
############################################################################



config_roundabout = rllib.basic.YamlConfig(
    mode=universe.AgentMode.interactive,
    case_ids=[None],
    map_name='roundabout',
    scenario_name='roundabout',
    dataset_dir=rllib.basic.BaseData(
        map=os.path.expanduser('~/dataset/carla/maps'),
    ),
    dataset_cls=dataset_cls,

    scenario_randomization_cls=scenarios_template.ScenarioRandomization,
    scenario_cls=scenarios_roundabout.ScenarioRoundabout,
    boundary=rllib.basic.BaseData(x_min=-20, y_min=-60, x_max=110, y_max=60),
    center=rllib.basic.BaseData(x=45, y=0),
    num_vehicles_range=rllib.basic.BaseData(min=8, max=20),
    # num_vehicles_range=rllib.basic.BaseData(min=1, max=1),  ### ! warning
    # num_vehicles_range=rllib.basic.BaseData(min=2, max=2),  ### ! warning

    decision_frequency=4, control_frequency=40,
    perception_range=50.0,

    bbx_x=config_vehicle.bbx_x, bbx_y=config_vehicle.bbx_y,
    config_vehicle=config_vehicle,

    agents_master_cls=agents_master.AgentListMaster,
    neural_vehicle_cls=agents_master.EndToEndVehicleWithCharacter,
    rule_vehicle_cls=agents_master.IdmVehicleWithCharacter,
    perception_cls=perception.PerceptionPointNet,
    reward_func=reward.RewardFunctionNoCharacter,

)


config_roundabout__with_character = copy.copy(config_roundabout)
config_roundabout__with_character.reward_func = reward.RewardFunctionWithCharacter

config_roundabout__with_character_share = copy.copy(config_roundabout__with_character)
config_roundabout__with_character_share.scenario_randomization_cls = scenarios_template.ScenarioRandomization_share_character

