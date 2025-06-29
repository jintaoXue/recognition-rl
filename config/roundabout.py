import rllib

import numpy as np
import os
import copy


import universe
from universe.carla.dataset import DatasetInteractive as dataset_cls
from utils import scenarios_template
from utils import scenarios_roundabout
from utils import agents_master
from utils import perception
from utils import reward



from config.bottleneck import config_vehicle



config_env = rllib.basic.YamlConfig(
    mode=universe.AgentMode.interactive,
    case_ids=[None],
    map_name='roundabout',
    scenario_name='roundabout',
    dataset_dir=rllib.basic.BaseData(
        map=os.path.expanduser('~/dataset/carla/maps'),
    ),
    dataset_cls=dataset_cls,

    scenario_randomization_cls=scenarios_template.ScenarioRandomizationRoundabout,
    scenario_cls=scenarios_roundabout.ScenarioRoundabout,
    boundary=rllib.basic.BaseData(x_min=-20, y_min=-60, x_max=110, y_max=60),
    center=rllib.basic.BaseData(x=45, y=0),
    num_vehicles_range=rllib.basic.BaseData(min=8, max=20),
    # num_vehicles_range=rllib.basic.BaseData(min=1, max=1),  ### ! warning
    # num_vehicles_range=rllib.basic.BaseData(min=2, max=2),  ### ! warning

    num_steps=100,
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


####################################################################################
### multi agent ####################################################################
####################################################################################


config_env__with_character = copy.copy(config_env)
config_env__with_character.reward_func = reward.RewardFunctionWithCharacter

config_env__with_character_share = copy.copy(config_env__with_character)
config_env__with_character_share.scenario_randomization_cls = scenarios_template.ScenarioRandomization_share_character




####################################################################################
### single agent ###################################################################
####################################################################################

config_env__neural_background = copy.copy(config_env)
config_env__neural_background.scenario_randomization_cls = scenarios_template.ScenarioRandomizationWithoutMismatch
config_env__neural_background.agents_master_cls = agents_master.AgentListMasterNeuralBackground
config_env__neural_background.rule_vehicle_cls = agents_master.EndToEndVehicleWithCharacterBackground


