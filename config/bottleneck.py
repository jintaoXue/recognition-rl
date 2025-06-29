import rllib

import numpy as np
import os
import copy


import universe
from universe.carla.dataset import DatasetInteractive as dataset_cls
from utils import scenarios_template
from utils import scenarios_bottleneck
from utils import agents_master
from utils import perception
from utils import reward
# from utils import topology_map

config_vehicle = rllib.basic.YamlConfig(
    min_velocity=0.0,
    max_velocity=6.0,
    max_acceleration=5.0,
    min_acceleration=-5.0,
    max_steer=np.deg2rad(45),

    wheelbase=2.6,
    # wheelbase=2.875,
    bbx_x=2.1, bbx_y=1.0,
    # bbx_x=2.395890, bbx_y=1.081725,

    idm_scale_x=1.0,
    idm_scale_y=1.6,

)


config_env = rllib.basic.YamlConfig(
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
    num_vehicles_range=rllib.basic.BaseData(min=20, max=20),
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

# config_env__with_character_fix_other_svo = copy.copy(config_env__with_character)
# config_env__with_character_fix_other_svo.scenario_randomization_cls = scenarios_template.ScenarioRandomizationWithoutMismatch

config_env__actsvo_multiagent = copy.copy(config_env)
config_env__actsvo_multiagent.agents_master_cls = agents_master.AgentListMasterActSvoMultiAgent


####################################################################################
### single agent ###################################################################
####################################################################################

config_env__neural_background = copy.copy(config_env)
config_env__neural_background.scenario_randomization_cls = scenarios_template.ScenarioRandomizationWithoutMismatch
config_env__neural_background.agents_master_cls = agents_master.AgentListMasterNeuralBackground
config_env__neural_background.rule_vehicle_cls = agents_master.EndToEndVehicleWithCharacterBackground

#ILAttn 
config_env__neural_background_mix = copy.copy(config_env)
config_env__neural_background_mix.scenario_cls = scenarios_bottleneck.ScenarioBottleneckDiverse
config_env__neural_background_mix.scenario_randomization_cls = scenarios_template.ScenarioRandomizationDivese
config_env__neural_background_mix.agents_master_cls = agents_master.AgentListMasterMixbackgrd
config_env__neural_background_mix.neural_vehicle_cls = agents_master.EndToEndVehicleDiverse
config_env__neural_background_mix.rule_vehicle_cls = agents_master.EndToEndVehicleBackgroundDiverse


config_env__neural_background_discrete_svo = copy.copy(config_env__neural_background)
config_env__neural_background_discrete_svo.scenario_randomization_cls = scenarios_template.ScenarioRandomizationWithoutMismatchDisSvo


# config_env__neural_background_sampling = copy.copy(config_env__neural_background)
# config_env__neural_background_sampling.perception_cls = perception_downsample.PerceptionPointNetDownSample
# config_env__neural_background_sampling.topology_map_cls = topology_map.TopologyMapSampled

config_env__neural_background_same_other_svo = copy.copy(config_env__neural_background)
# config_env__neural_background_same_other_svo.scenario_randomization_cls = scenarios_template.ScenarioRandomizationFixOtherSvo
# config_env__neural_background_same_other_svo.reward_func = reward.RewardFunctionRecogCharacter

config_env__new_action_background = copy.copy(config_env__neural_background_same_other_svo)
config_env__new_action_background.agents_master_cls = agents_master.AgentListMasterNeuralBackgroundRecog
config_env__new_action_background.reward_func = reward.RewardFunctionRecogCharacterV1

config_env__new_action_multi_svo = copy.copy(config_env__neural_background_same_other_svo)
config_env__new_action_multi_svo.agents_master_cls = agents_master.AgentListMasterNeuralBackgroundRecogMultiSVO
config_env__new_action_multi_svo.reward_func = reward.RewardFunctionRecogCharacterV2

config_env__multiact__mixbkgrd = copy.copy(config_env)
# config_env__multiact__mixbkgrd.scenario_randomization_cls = scenarios_template.ScenarioRandomizationWithoutMismatch
config_env__multiact__mixbkgrd.scenario_cls = scenarios_bottleneck.ScenarioBottleneckDiverse
config_env__multiact__mixbkgrd.scenario_randomization_cls = scenarios_template.ScenarioRandomizationDivese
config_env__multiact__mixbkgrd.agents_master_cls = agents_master.AgentListMasterSVOasActMixbackgrd
config_env__multiact__mixbkgrd.neural_vehicle_cls = agents_master.EndToEndVehicleDiverse
config_env__multiact__mixbkgrd.rule_vehicle_cls = agents_master.EndToEndVehicleBackgroundDiverse
