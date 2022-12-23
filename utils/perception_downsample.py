import rllib
import universe

import numpy as np

import matplotlib.pyplot as plt
from typing import List


from universe.common.topology_map import transform_poses, transform_points

from .perception import PerceptionPointNet, PerceptionVectorizedMap, PerceptionVectorizedRoute



class PerceptionPointNetDownSample(PerceptionPointNet):
    def __init__(self, config, topology_map: universe.common.TopologyMap, dim_vehicle_state, horizon):
        self.config = config
        self.decision_frequency = config.decision_frequency
        self.perception_range = config.perception_range
        self.horizon = horizon


        # self.num_vehicles = num_vehicles
        self.num_agents_max = self.config.num_vehicles_range.max  ### ! warning


        self.historical_timestamps = -np.array([range(self.horizon)], dtype=np.float32).T[::-1] /self.decision_frequency
        sampling_resolution = 2.0

        self.perp_route = PerceptionVectorizedRouteDownSample(config, sampling_resolution=sampling_resolution)
        self.perp_map = PerceptionVectorizedMapDownSample(config, topology_map)


        # self.dim_state = rllib.basic.BaseData(agent=dim_vehicle_state, static=self.perp_map.dim_state)


        '''viz'''
        self.default_colors = rllib.basic.Data(
            ego='r', obs='g', route='b',
            lane='#D3D3D3',  ### lightgray
            bound='#800080', ### purple
        )
        return


# interval = 1
# interval = 2
interval = 4

class PerceptionVectorizedRouteDownSample(PerceptionVectorizedRoute):
    dim_state = 4
    num_points = 20

    def __init__(self, config, sampling_resolution=2.0):
        self.perception_range = config.perception_range
        self.sampling_resolution = sampling_resolution


    def run_step(self, vehicle: universe.common.Vehicle):
        state0 = vehicle.get_state()
        global_path = vehicle.global_path
        global_path.step_coverage(vehicle.get_transform())
        s = global_path.max_coverage
        i = np.clip(int(self.sampling_resolution / global_path.sampling_resolution), 1, None)
        d = self.num_points
        x, y, theta = global_path.x[s::i][:d], global_path.y[s::i][:d], global_path.theta[s::i][:d]
        x, y, theta = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(theta, dtype=np.float32)
        spacestamps = np.arange(0, self.num_points, dtype=np.float32)/ self.num_points
        route = np.stack([x, y, theta, spacestamps[:len(x)]], axis=1)
        route = np.concatenate([
            route,
            np.full([self.num_points-route.shape[0], route.shape[1]], np.inf, dtype=np.float32),
        ], axis=0)
        route_mask = np.where(route < np.inf, 1,0).all(axis=-1)
        route = transform_poses(route, state0)

        ### normalize
        route[:,:2] /= self.perception_range
        route[:,2] /= np.pi

        return rllib.basic.Data(route=route, route_mask=route_mask)[::interval]





class PerceptionVectorizedMapDownSample(PerceptionVectorizedMap):
    invalid_value = np.inf
    dim_state = 4

    def __init__(self, config, topology_map: universe.common.TopologyMap):
        self.config = config
        self.topology_map = topology_map
        self.perception_range = config.perception_range
        return


    def run_step(self, step_reset, time_step, agent: universe.common.EndToEndVehicle):
        self.step_reset, self.time_step = step_reset, time_step
        state0 = agent.get_state()

        lane, lane_mask = self.topology_map.crop_line(state0, self.perception_range, line_type='center')
        bound, bound_mask = self.topology_map.crop_line(state0, self.perception_range, line_type='side')

        bound_flag = self.check_in_bound(agent, bound)

        ### normalize
        lane[:,:,:3] /= self.perception_range
        lane[:,:,3] /= 50
        bound[:,:,:3] /= self.perception_range
        bound[:,:,3] /= 50

        it = interval  ### interval
        return rllib.basic.Data(lane=lane[:,::it].copy(), lane_mask=lane_mask[:,::it].copy(), bound=bound[:,::it].copy(), bound_mask=bound_mask[:,::it].copy(), bound_flag=bound_flag)


