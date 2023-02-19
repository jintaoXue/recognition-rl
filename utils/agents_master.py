from matplotlib.pyplot import axis
from pyparsing import actions
from pyrsistent import b
import rllib

import numpy as np
import copy
import torch

import universe
from universe.common.color import ColorLib
from universe.common.vehicle import Vehicle
from universe.common.vehicle_neural import NeuralVehicle,EndToEndVehicle
from universe.common.vehicle_rule import IdmVehicle, RuleVehicle
from universe.common.topology_map import TopologyMap
from universe.perception import Perception
from typing import List

class EndToEndVehicleWithCharacter(EndToEndVehicle):
    def __init__(self, config, vi, global_path, transform, character):
        super().__init__(config, vi, global_path, transform)
        self.character = character

    def __str__(self):
        state = copy.deepcopy(self.get_state())
        x = str(round(state.x, 2))
        y = str(round(state.y, 2))
        character = str(np.round(self.character, 2))
        return str(type(self))[:-1] + f' vi: {self.vi}, x: {x}, y: {y}, svo: {character}>'

#the vehicle with diverse type such as flow, robust 
#thus the vehicle is controll by different policy in a same scenario
class EndToEndVehicleDiverse(EndToEndVehicleWithCharacter) :
    # type = flow, robust
    def __init__(self, config, vi, global_path, transform, character, control_type):
        super().__init__(config, vi, global_path, transform, character)
        self.control_type = control_type

class IdmVehicleWithCharacter(IdmVehicle):
    def __init__(self, config, vi, global_path, transform, character):
        super().__init__(config, vi, global_path, transform)
        self.character = character

    def __str__(self):
        state = copy.deepcopy(self.get_state())
        x = str(round(state.x, 2))
        y = str(round(state.y, 2))
        v = str(round(state.v, 2))
        leading_vi = None
        if hasattr(self, 'leading_vehicle') and self.leading_vehicle != None:
            leading_vi = self.leading_vehicle.vi
        return str(type(self))[:-1] + f' vi: {self.vi}, x: {x}, y: {y}, v: {v}, leading: {leading_vi}>'

class EndToEndVehicleWithCharacterBackground(RuleVehicle):
    dim_action = 2

    def __init__(self, config, vi, global_path, transform, character):
        super().__init__(config, vi, global_path, transform)
        self.character = character

    def set_svo(self, reference):
        character = np.clip(reference[0], -1,1)
        self.character = ((character + 1) *0.5).astype(np.float32)
        return


    def get_target(self, reference):
        acc = np.clip(reference[0].item(), -1,1) * self.max_acceleration
        steer = np.clip(reference[1].item(), -1,1) * self.max_steer
        return universe.common.VehicleControl(acc, steer)


    def get_control(self, target):
        control = target
        return control


    def __str__(self):
        state = copy.deepcopy(self.get_state())
        x = str(round(state.x, 2))
        y = str(round(state.y, 2))
        v = str(round(state.v, 2))
        character = str(np.round(self.character, 2))
        return str(type(self))[:-1] + f' vi: {self.vi}, x: {x}, y: {y}, v: {v}, svo: {character}>'

class EndToEndVehicleBackgroundDiverse(EndToEndVehicleWithCharacterBackground):
    # type = flow, robust
    def __init__(self, config, vi, global_path, transform, character, control_type):
        super().__init__(config, vi, global_path, transform, character)
        self.control_type = control_type
    def __str__(self):
        state = copy.deepcopy(self.get_state())
        x = str(round(state.x, 2))
        y = str(round(state.y, 2))
        v = str(round(state.v, 2))
        character = str(np.round(self.character, 2))
        control_type = str(self.control_type)
        return str(type(self))[:-1] + f' vi: {self.vi}, x: {x}, y: {y}, v: {v}, svo: {character}, control_type:{control_type}>'


class AgentListMaster(universe.AgentsMaster):
    # change in 2022/12/2
    def __init__(self, config: rllib.basic.YamlConfig, topology_map: TopologyMap, **kwargs):
        super().__init__(config, topology_map, **kwargs)
        self.horizon = config.raw_horizon
        perception_cls = config.get('perception_cls', Perception)
        self.perception = perception_cls(config, topology_map, self.dim_vehicle_state, self.horizon)
    class GetVehicleState(object):
        dim_state = 5
        def run_step(self, vehicle: Vehicle):
            state = vehicle.get_state()
            return np.array([
                state.x, state.y, state.theta, state.v,
                vehicle.character,
            ], dtype=np.float32)


class AgentListMasterNeuralBackground(AgentListMaster):
    """
        Only for single agent.
    """
    
    def __init__(self, config: rllib.basic.YamlConfig, topology_map, **kwargs):
        super().__init__(config, topology_map, **kwargs)

        config_neural_policy = config.config_neural_policy
        config_neural_policy.set('dim_state', config.dim_state)
        config_neural_policy.set('dim_action', config.dim_action)
        self.buffer_cls = config_neural_policy.buffer
        self.device = config_neural_policy.device
        self.neural_policy = rllib.sac.Actor(config_neural_policy).to(self.device)
        self.neural_policy.load_model()

    def observe(self, step_reset, time_step):
        if time_step == 0:
            for vehicle in self.vehicles_neural + self.vehicles_rule:
                self.vehicle_states[vehicle.vi, :self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
                self.vehicle_masks[vehicle.vi, :self.horizon-1] = 1
        
        for vehicle in self.vehicles_neural + self.vehicles_rule:
            self.vehicle_states[vehicle.vi, time_step + self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
            self.vehicle_masks[vehicle.vi, time_step + self.horizon-1] = 3

        vehicle_states = self.vehicle_states[:, time_step:time_step+self.horizon]
        vehicle_masks = self.vehicle_masks[:, time_step:time_step+self.horizon]

        state = self.perception.run_step(step_reset, time_step, self.vehicles_neural + self.vehicles_rule, vehicle_states, vehicle_masks)
        self.state_neural_nackground = state[1:]
        return [state[0]]

    def run_step(self, references):
        """
        Args:
            references: torch.Size([num_agents_learnable, dim_action])
        """
        assert len(references) == len(self.vehicles_neural)
        state_bg = [s.to_tensor().unsqueeze(0) for s in self.state_neural_nackground]
        if len(state_bg) > 0:
            states_bg = rllib.buffer.stack_data(state_bg)
            self.buffer_cls.pad_state(None, states_bg)
            states_bg = states_bg.cat(dim=0)
            actions_bg, _, _ = self.neural_policy.sample(states_bg.to(self.device))
            actions_bg = actions_bg.detach().cpu().numpy()
        else:
            actions_bg = []
        
        vehicles = self.vehicles_neural + self.vehicles_rule
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_rule = [vehicle.get_target(action_bg) for vehicle, action_bg in zip(self.vehicles_rule, actions_bg)]
        targets = targets_neural + targets_rule
        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return
    

class AgentListMasterNeuralBackgroundManualTuneSVOCoPO(AgentListMasterNeuralBackground):
    """
        Only for multi agent without vehicles_rule.
    """
    

    def run_step(self, references):
        """
        Args:
            references: torch.Size([num_agents_learnable, dim_action])
        """

        assert len(references) == len(self.vehicles_neural)

        if self.config.scenario_name == 'bottleneck':
            svos = np.random.normal(0.8, 0.05, size=(len(self.vehicles_rule), 1))
            svos = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'intersection_v2':
            svos = np.random.normal(0.9, 0.1, size=(len(self.vehicles_rule), 1))
            svos = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'merge':
            svos = np.random.normal(0.5, 0.05, size=(len(self.vehicles_rule), 1))
            svos = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'roundabout':
            svos = np.random.normal(0.8, 0.05, size=(len(self.vehicles_rule), 1))
            svos = (svos *2 -1).astype(np.float32)
        else:
            raise NotImplementedError(f'unkonwn {self.config.scenario_name}')
        [vehicle.set_svo(reference) for vehicle, reference in zip(self.vehicles_rule, svos)]

        state_bg = [s.to_tensor().unsqueeze(0) for s in self.state_neural_nackground]
        if len(state_bg) > 0:
            states_bg = rllib.buffer.stack_data(state_bg)
            self.buffer_cls.pad_state(None, states_bg)
            states_bg = states_bg.cat(dim=0)
            actions_bg, _, _ = self.neural_policy.sample(states_bg.to(self.device))
            actions_bg = actions_bg.detach().cpu().numpy()
        else:
            actions_bg = []

        vehicles = self.vehicles_neural + self.vehicles_rule
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_rule = [vehicle.get_target(action_bg) for vehicle, action_bg in zip(self.vehicles_rule, actions_bg)]
        targets = targets_neural + targets_rule

        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return


class AgentListMasterNeuralBackgroundManualTuneSVOCoPOAdv(AgentListMasterNeuralBackground):
    """
        Only for multi agent without vehicles_rule.
    """
    

    def run_step(self, references):
        """
        Args:
            references: torch.Size([num_agents_learnable, dim_action])
        """

        assert len(references) == len(self.vehicles_neural)

        if self.config.scenario_name == 'bottleneck':
            svos = np.random.normal(0.0, 0.1, size=(len(self.vehicles_rule), 1))
            svos = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'intersection_v2':
            svos = np.random.normal(0.0, 0.05, size=(len(self.vehicles_rule), 1))
            svos = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'merge':
            svos = np.random.normal(1.0, 0.05, size=(len(self.vehicles_rule), 1))
            svos = (svos *2 -1).astype(np.float32)
        elif self.config.scenario_name == 'roundabout':
            svos = np.random.normal(1.0, 0.05, size=(len(self.vehicles_rule), 1))
            svos = (svos *2 -1).astype(np.float32)
        else:
            raise NotImplementedError(f'unkonwn {self.config.scenario_name}')
        [vehicle.set_svo(reference) for vehicle, reference in zip(self.vehicles_rule, svos)]

        state_bg = [s.to_tensor().unsqueeze(0) for s in self.state_neural_nackground]
        if len(state_bg) > 0:
            states_bg = rllib.buffer.stack_data(state_bg)
            self.buffer_cls.pad_state(None, states_bg)
            states_bg = states_bg.cat(dim=0)
            actions_bg, _, _ = self.neural_policy.sample(states_bg.to(self.device))
            actions_bg = actions_bg.detach().cpu().numpy()
        else:
            actions_bg = []

        vehicles = self.vehicles_neural + self.vehicles_rule
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_rule = [vehicle.get_target(action_bg) for vehicle, action_bg in zip(self.vehicles_rule, actions_bg)]
        targets = targets_neural + targets_rule

        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return

class AgentListMasterNeuralBackgroundRecog(AgentListMaster):
    """
        Only for single agent.
    """
    def __init__(self, config: rllib.basic.YamlConfig, topology_map, **kwargs):
        super().__init__(config, topology_map, **kwargs)

        config_neural_policy = config.config_neural_policy
        config_neural_policy.set('dim_state', config.dim_state)
        config_neural_policy.set('dim_action', config.dim_action)
        self.buffer_cls = config_neural_policy.buffer
        self.device = config_neural_policy.device
        from core.recognition_net import Actor
        self.neural_policy = Actor(config_neural_policy).to(self.device)
        self.neural_policy.load_model()
        self.state = rllib.basic.data.Data()

    def observe(self, step_reset, time_step):
        if time_step == 0:
            for vehicle in self.vehicles_neural + self.vehicles_rule:
                self.vehicle_states[vehicle.vi, :self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
                self.vehicle_masks[vehicle.vi, :self.horizon-1] = 1
        
        for vehicle in self.vehicles_neural + self.vehicles_rule:
            self.vehicle_states[vehicle.vi, time_step + self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
            self.vehicle_masks[vehicle.vi, time_step + self.horizon-1] = 3

        vehicle_states = self.vehicle_states[:, time_step:time_step+self.horizon]
        vehicle_masks = self.vehicle_masks[:, time_step:time_step+self.horizon]

        state = self.perception.run_step(step_reset, time_step, self.vehicles_neural + self.vehicles_rule, vehicle_states, vehicle_masks)
        self.state_neural_nackground = state[1:]
        self.state = state[0]
        # breakpoint()
        return [state[0]]

    def run_step(self, obs_svo):
        """
        Args:
            obs_svo: torch.Size([num_agents_learnable, 1])
        """
        # assert len(obs_svo) == len(self.vehicles_neural)
        #froms
        obs_svo = torch.from_numpy(obs_svo).to(self.device)
        obs_svos = obs_svo.unsqueeze(1).repeat(1,len(self.state.obs_character),1)
        # obs_svos = np.expand_dims(obs_svo, axis = 1).repeat(len(self.state.obs_character),axis=1)
        state = self.state.to_tensor().unsqueeze(0)
        references= self.neural_policy.forward_with_svo(state.to(self.device), obs_svos.to(self.device))
        state_bg = [s.to_tensor().unsqueeze(0) for s in self.state_neural_nackground]

        if len(state_bg) > 0:
            states_bg = rllib.buffer.stack_data(state_bg)
            self.buffer_cls.pad_state(None, states_bg)
            states_bg = states_bg.cat(dim=0)
            actions_bg, _, _ = self.neural_policy.sample(states_bg.to(self.device))
            actions_bg = actions_bg.detach().cpu().numpy()
        else:
            actions_bg = []
        
        vehicles = self.vehicles_neural + self.vehicles_rule
        references = references.detach().cpu().numpy()
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_rule = [vehicle.get_target(action_bg) for vehicle, action_bg in zip(self.vehicles_rule, actions_bg)]
        targets = targets_neural + targets_rule
        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return

class AgentListMasterNeuralBackgroundRecogMultiSVO(AgentListMasterNeuralBackgroundRecog):
    """
        Only for single agent.
    """
    def run_step(self, obs_svos):
        """
        Args:
            obs_svos: torch.Size([batch, self.max_obs_vehicle])
        """
        # assert len(obs_svos) == len(self.vehicles_neural)
        #froms

        obs_svos = obs_svos[:,:len(self.state.obs_character)]
        obs_svos = torch.from_numpy(obs_svos).to(self.device)
        # obs_svos = np.expand_dims(obs_svos, axis=-1)
        state = self.state.to_tensor().unsqueeze(0)
        # print('agent master obs_svos', obs_svos.shape)
        references = self.neural_policy.forward_with_svo(state.to(self.device), obs_svos.to(self.device))
        state_bg = [s.to_tensor().unsqueeze(0) for s in self.state_neural_nackground]

        if len(state_bg) > 0:
            states_bg = rllib.buffer.stack_data(state_bg)
            self.buffer_cls.pad_state(None, states_bg)
            states_bg = states_bg.cat(dim=0)
            actions_bg, _, _ = self.neural_policy.sample(states_bg.to(self.device))
            actions_bg = actions_bg.detach().cpu().numpy()
        else:
            actions_bg = []
        
        vehicles = self.vehicles_neural + self.vehicles_rule
        references = references.detach().cpu().numpy()
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_rule = [vehicle.get_target(action_bg) for vehicle, action_bg in zip(self.vehicles_rule, actions_bg)]
        targets = targets_neural + targets_rule
        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return



###########for mix agent including copo, flow

class AgentListMasterMixbackgrd(AgentListMaster):
    def __init__(self, config: rllib.basic.YamlConfig, topology_map, **kwargs):
        super().__init__(config, topology_map, **kwargs)
        ##ours for ego agent 
        config_neural_policy_ours = config.config_neural_policy_ours
        config_neural_policy_ours.set('dim_state', config.dim_state)
        config_neural_policy_ours.set('dim_action', config.dim_action)
        ##robust
        config_neural_policy_robust = config.config_neural_policy_robust
        config_neural_policy_robust.set('dim_state', config.dim_state)
        config_neural_policy_robust.set('dim_action', config.dim_action)
        #flow
        config_neural_policy_flow = config.config_neural_policy_flow
        config_neural_policy_flow.set('dim_state', config.dim_state)
        config_neural_policy_flow.set('dim_action', config.dim_action)

        # core.model_vectornet.ReplayBufferMultiAgentMultiWorker -> buffer 
        self.buffer_cls = config_neural_policy_robust.buffer
        self.device = config_neural_policy_robust.device

        #policy
        # from core.recognition_net import Actor
        from core.recognition_net import Actor
        self.neural_policy_ours = Actor(config_neural_policy_ours).to(self.device)
        self.neural_policy_robust = rllib.sac.Actor(config_neural_policy_robust).to(self.device)
        self.neural_policy_flow = rllib.sac.Actor(config_neural_policy_flow).to(self.device)
        self.neural_policy_ours.load_model()
        self.neural_policy_robust.load_model()
        self.neural_policy_flow.load_model()
        self.control_types_num = {
            "robust": 0,
            "flow": 0,
            "idm": 0, 
            }
        self.state = rllib.basic.data.Data()

    def observe(self, step_reset, time_step):
        if time_step == 0:
            for vehicle in self.vehicles_neural + self.vehicles_rule:
                self.vehicle_states[vehicle.vi, :self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
                self.vehicle_masks[vehicle.vi, :self.horizon-1] = 1
        
        for vehicle in self.vehicles_neural + self.vehicles_rule:
            self.vehicle_states[vehicle.vi, time_step + self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
            self.vehicle_masks[vehicle.vi, time_step + self.horizon-1] = 3

        vehicle_states = self.vehicle_states[:, time_step:time_step+self.horizon]
        vehicle_masks = self.vehicle_masks[:, time_step:time_step+self.horizon]

        state = self.perception.run_step(step_reset, time_step, self.vehicles_neural + self.vehicles_rule, vehicle_states, vehicle_masks)
        self.state_neural_nackground = state[1:]
        self.state = state[0]
        # breakpoint()
        return [state[0]]

    def get_action_bg(self, states_bg):
        index_0 = self.control_types_num['robust']
        index_1 = self.control_types_num['robust'] + self.control_types_num['flow']
        actions_bg_robust, _, _ = self.neural_policy_robust.sample(states_bg[0:index_0].to(self.device))
        actions_bg_flow,_,_ = self.neural_policy_flow.sample(states_bg[index_0:index_1].to(self.device))
        return torch.cat([actions_bg_robust,actions_bg_flow], axis=0)

    def run_step(self, references):
        """
        Args:
            references: torch.Size([num_agents_learnable, dim_action])
        """
        # assert len(obs_svos) == len(self.vehicles_neural)
        #froms
        state_bg = [s.to_tensor().unsqueeze(0) for s in self.state_neural_nackground]

        if len(state_bg) > 0:
            states_bg = rllib.buffer.stack_data(state_bg)
            self.buffer_cls.pad_state(None, states_bg)
            states_bg = states_bg.cat(dim=0)
            actions_bg =  self.get_action_bg(states_bg)
            # actions_bg, _, _ = self.neural_policy.sample(states_bg.to(self.device))
            # actions_bg = actions_bg.detach().cpu().numpy()
        else:
            actions_bg = []
        
        vehicles = self.vehicles_neural + self.vehicles_rule
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_rule = [vehicle.get_target(action_bg) for vehicle, action_bg in zip(self.vehicles_rule, actions_bg)]
        targets = targets_neural + targets_rule
        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return
    


class AgentListMasterSVOasActMixbackgrd(AgentListMasterMixbackgrd):
    def run_step(self, obs_svos):
        """
        Args:
            obs_svos: torch.Size([batch, self.max_obs_vehicle])
        """
        # assert len(obs_svos) == len(self.vehicles_neural)
        #froms

        obs_svos = obs_svos[:,:len(self.state.obs_character)]
        obs_svos = torch.from_numpy(obs_svos).to(self.device)
        # obs_svos = np.expand_dims(obs_svos, axis=-1)
        state = self.state.to_tensor().unsqueeze(0)
        # print('agent master obs_svos', obs_svos.shape)
        references = self.neural_policy_ours.forward_with_svo(state.to(self.device), obs_svos.to(self.device))
        state_bg = [s.to_tensor().unsqueeze(0) for s in self.state_neural_nackground]

        if len(state_bg) > 0:
            states_bg = rllib.buffer.stack_data(state_bg)
            self.buffer_cls.pad_state(None, states_bg)
            states_bg = states_bg.cat(dim=0)
            actions_bg =  self.get_action_bg(states_bg)
            # actions_bg, _, _ = self.neural_policy.sample(states_bg.to(self.device))
            # actions_bg = actions_bg.detach().cpu().numpy()
        else:
            actions_bg = []
        
        vehicles = self.vehicles_neural + self.vehicles_rule
        references = references.detach().cpu().numpy()
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets_rule = [vehicle.get_target(action_bg) for vehicle, action_bg in zip(self.vehicles_rule, actions_bg)]
        targets = targets_neural + targets_rule
        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return

class AgentListMasterActSvoMultiAgent(AgentListMaster) :
    """
        Only for multi agent without vehicles_rule.
    """
    def __init__(self, config: rllib.basic.YamlConfig, topology_map, **kwargs):
        super().__init__(config, topology_map, **kwargs)

        config_neural_policy = config.config_neural_policy
        config_neural_policy.set('dim_state', config.dim_state)
        config_neural_policy.set('dim_action', config.dim_action)
        self.buffer_cls = config_neural_policy.buffer
        self.device = config_neural_policy.device
        from core.recognition_net import Actor
        self.neural_policy = Actor(config_neural_policy).to(self.device)
        self.neural_policy.load_model()
    
    def observe(self, step_reset, time_step):
        if time_step == 0:
            for vehicle in self.vehicles_neural + self.vehicles_rule:
                self.vehicle_states[vehicle.vi, :self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
                self.vehicle_masks[vehicle.vi, :self.horizon-1] = 1
        
        for vehicle in self.vehicles_neural + self.vehicles_rule:
            self.vehicle_states[vehicle.vi, time_step + self.horizon-1] = self.get_vehicle_state.run_step(vehicle)
            self.vehicle_masks[vehicle.vi, time_step + self.horizon-1] = 3

        vehicle_states = self.vehicle_states[:, time_step:time_step+self.horizon]
        vehicle_masks = self.vehicle_masks[:, time_step:time_step+self.horizon]

        state = self.perception.run_step(step_reset, time_step, self.vehicles_neural, vehicle_states, vehicle_masks)
        self.state = state
        return state

    def run_step(self, obs_svos, state):
        """
        Args:
            obs_svos: torch.Size([batch, self.max_obs_vehicle])
        """
        # assert len(obs_svos) == len(self.vehicles_neural)
        #froms
        state = [s.to_tensor().unsqueeze(0) for s in state]
        state = rllib.buffer.stack_data(state)
        # state = self.buffer_cls._batch_stack(self.state)
        self.buffer_cls.pad_state(None, state)
        state = state.cat(dim=0)
        obs_svos = obs_svos[:,:state.obs_character.shape[1]]
        obs_svos = torch.from_numpy(obs_svos).to(self.device)
        # obs_svos = np.expand_dims(obs_svos, axis=-1)
        # print('agent master obs_svos', obs_svos.shape)
        references = self.neural_policy.forward_with_svo(state.to(self.device), obs_svos)
        
        vehicles = self.vehicles_neural + self.vehicles_rule
        references = references.detach().cpu().numpy()
        targets_neural = [vehicle.get_target(reference) for vehicle, reference in zip(self.vehicles_neural, references)]
        targets = targets_neural 
        for _ in range(self.skip_num):
            for vehicle, target in zip(vehicles, targets):
                control = vehicle.get_control(target)
                vehicle.forward(control)
        for vehicle in vehicles:
            vehicle.tick()
        return
