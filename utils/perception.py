import imp
from operator import index
from random import sample
from turtle import color
from cv2 import sampsonDistance
import rllib
from torch import float32
import universe

import numpy as np

import matplotlib.pyplot as plt
from typing import List


from universe.common.topology_map import transform_poses, transform_points
from universe.common.geo import Transform
from universe.common.global_path import calc_curvature_with_yaw_diff

class PerceptionPointNet(object):
    invalid_value = np.inf

    dim_state = rllib.basic.BaseData(agent=5, static=4)  ### ! warning todo


    def __init__(self, config, topology_map: universe.common.TopologyMap, dim_vehicle_state, horizon):
        self.config = config
        self.decision_frequency = config.decision_frequency
        self.perception_range = config.perception_range
        self.horizon = horizon


        # self.num_vehicles = num_vehicles
        self.num_agents_max = self.config.num_vehicles_range.max  ### ! warning


        self.historical_timestamps = -np.array([range(self.horizon)], dtype=np.float32).T[::-1] /self.decision_frequency
        sampling_resolution = 4.13
        '''route resolution before sampling: 2.059351921081543, after sampling: 2.0951883792877197'''
        # sampling_resolution = 2.0
        '''route resolution before sampling: 2.059351921081543, after sampling: 4.252692222595215'''

        self.perp_route = PerceptionVectorizedRoute(config, sampling_resolution=sampling_resolution)
        self.perp_map = PerceptionVectorizedMap(config, topology_map)


        # self.dim_state = rllib.basic.BaseData(agent=dim_vehicle_state, static=self.perp_map.dim_state)


        '''viz'''
        self.default_colors = rllib.basic.Data(
            ego='r', obs='g', route='b',
            lane='#D3D3D3',  ### lightgray
            bound='#800080', ### purple
        )
        return




    def run_step(self, step_reset, time_step, agents: List[universe.common.EndToEndVehicle], vehicle_states, vehicle_masks):
        self.step_reset, self.time_step = step_reset, time_step

        states = []
        for agent in agents:
            state_vehicle = self.get_vehicles(agent, vehicle_states, vehicle_masks)
            state_route = self.perp_route.run_step(agent)
            state_map = self.perp_map.run_step(step_reset, time_step, agent)

            ### only for multi-agent
            agent_state = agent.get_state()
            agent_states = vehicle_states[:,-1]
            dist = np.sqrt((agent_states[:,0]-agent_state.x)**2 + (agent_states[:,1]-agent_state.y)**2)
            agent_masks = np.where(dist < agent.perception_range, 1, 0)

            state = rllib.basic.Data(step_reset=step_reset, time_step=time_step, vi=agent.vi, 
                agent_masks=agent_masks, vehicle_masks=vehicle_masks[:,-1]) + state_vehicle + state_route + state_map
            states.append(state)
        return states



    def get_vehicles(self, agent: universe.common.EndToEndVehicle, vehicle_states, vehicle_masks):
        ego_states = vehicle_states[agent.vi]
        ego_masks = vehicle_masks[agent.vi]
        other_states = np.delete(vehicle_states, agent.vi, axis=0)
        other_masks = np.delete(vehicle_masks, agent.vi, axis=0)  ## for multi-agent

        state0 = agent.get_state()

        # assert (state0.numpy() == ego_states[-1]).all()

        ego_states = transform_poses(ego_states, state0)
        other_states = transform_poses(other_states, state0)

        dist = np.sqrt(other_states[...,0]**2 + other_states[...,1]**2)
        valid_masks = np.where(dist < self.perception_range, 1, 0)
        valid_lengths = valid_masks.sum(axis=1)

        _valid_masks_union = np.expand_dims(valid_masks, axis=2).repeat(other_states.shape[2], axis=2)
        valid_other_states = np.where(_valid_masks_union, other_states, self.invalid_value)
        valid_other_states = np.delete(valid_other_states, np.where(valid_lengths == 0)[0], axis=0)
        valid_masks = np.delete(valid_masks, np.where(valid_lengths == 0)[0], axis=0)

        ego_ht = self.historical_timestamps
        ego_states = np.concatenate([ego_states, ego_ht], axis=1)
        other_ht = np.expand_dims(self.historical_timestamps, axis=0).repeat(valid_other_states.shape[0], axis=0)
        valid_other_states = np.concatenate([valid_other_states, other_ht], axis=2)

        ### left align
        sorted_index = np.expand_dims(np.argsort(-valid_masks[:,-1], axis=0, kind='mergesort'), axis=1).repeat(valid_masks.shape[1],axis=1)
        valid_masks = np.take_along_axis(valid_masks, sorted_index, axis=0)
        valid_other_states = np.take_along_axis(valid_other_states, np.expand_dims(sorted_index, axis=2).repeat(valid_other_states.shape[2], axis=2), axis=0)
 
        ### normalize
        ego_states[:, :2] /= self.perception_range
        ego_states[:, 2] /= np.pi
        ego_states[:, 3] /= agent.max_velocity
        ego_states[:, -1] /= (self.horizon/self.decision_frequency)

        valid_other_states[:,:, :2] /= self.perception_range
        valid_other_states[:,:, 2] /= np.pi
        valid_other_states[:,:, 3] /= agent.max_velocity
        valid_other_states[:,:, -1] /= (self.horizon/self.decision_frequency)

        ### split state and character
        ego_character = agent.character
        ego_states = np.delete(ego_states, -2, axis=-1)
        other_characters = valid_other_states[...,[-2]]
        valid_other_states = np.delete(valid_other_states, -2, axis=-1)

        return rllib.basic.Data(ego=ego_states, ego_mask=ego_masks, character=ego_character, obs=valid_other_states, obs_mask=valid_masks, obs_character=other_characters)




    def render(self, ax, data: List[rllib.basic.Data], colors=None):
        raise NotImplementedError

    def render_vi(self, ax, data: rllib.basic.Data, colors=None):
        print(rllib.basic.prefix(self) + 'viz')
        ax.clear()

        if colors == None:
            colors = rllib.basic.Data(
                ego=self.default_colors.ego,
                obs=[self.default_colors.obs] *len(data.obs),
                route=self.default_colors.route,
                lane=[self.default_colors.lane] *len(data.lane),
                bound=[self.default_colors.bound] *len(data.bound),
            )



        ### 0. perception circle
        circle_x, circle_y = 0, 0
        circle_r = self.perception_range /self.perception_range
        a_x = np.arange(0, 2*np.pi, 0.01)
        a = circle_x + circle_r *np.cos(a_x)
        b = circle_y + circle_r *np.sin(a_x)
        ax.plot(a, b, '-b')
        ax.plot(a,-b, '-b')

        ### 1.1 lane
        for lane, c in zip(data.lane, colors.lane):
            lane_x, lane_y = lane[:,0], lane[:,1]
            ax.plot(lane_x, lane_y, '-', color=c)

        ### 1.2 route
        route_x, route_y = data.route[:,0], data.route[:,1]
        ax.plot(route_x, route_y, '-', color=colors.route)

        ### 1.3 bound
        for bound, c in zip(data.bound, colors.bound):
            bound_x, bound_y = bound[:,0], bound[:,1]
            ax.plot(bound_x, bound_y, '-', color=c)

        ### 3. obs
        for obs, c in zip(data.obs, colors.obs):
            obs_x, obs_y = obs[:,0], obs[:,1]
            ax.plot(obs_x, obs_y, '-o', color=c)


        ### 4. ego
        ego_x, ego_y = data.ego[:,0], data.ego[:,1]
        ax.plot(ego_x, ego_y, '-o', color=colors.ego)


        # ax.savefig('{}.png'.format('results/tmp/'+str(self.time_step)))
        # ax.show()

        # import os
        # save_dir = os.path.join(self.config.path_pack.output_path, str(data.step_reset))
        # cu.system.mkdir(save_dir)
        # print(rllib.basic.prefix(self) + 'save fig')
        # ax.savefig( os.path.join(save_dir, '{}.png'.format(self.time_step)) )
        return



class PerceptionPointNetAdaptiveResolution(PerceptionPointNet):

    def __init__(self, config, topology_map: universe.common.TopologyMap, dim_vehicle_state, horizon):

        self.config = config
        self.decision_frequency = config.decision_frequency
        self.perception_range = config.perception_range
        self.horizon = horizon
        self.sampling_resolution = 4.0
        self.sampling_curvature_up_bound = 0.1
        
        # self.num_vehicles = num_vehicles
        self.num_agents_max = self.config.num_vehicles_range.max  ### ! warning


        self.historical_timestamps = -np.array([range(self.horizon)], dtype=np.float32).T[::-1] /self.decision_frequency

        self.perp_route = PerceptionVectorizedRoute(config, sampling_resolution=self.sampling_resolution)
        self.perp_map = MapAdaptiveResolution(config, topology_map, self.sampling_resolution, self.sampling_curvature_up_bound)


        # self.dim_state = rllib.basic.BaseData(agent=dim_vehicle_state, static=self.perp_map.dim_state)


        '''viz'''
        self.default_colors = rllib.basic.Data(
            ego='r', obs='g', route='b',
            lane='#D3D3D3',  ### lightgray
            bound='#800080', ### purple
        )
        return 


class PerceptionVectorizedRoute(object):
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
        #debug
        # route_ = np.expand_dims(route, axis=0)
        # print("route resolution before sampling: {}, after sampling: {}\n".format(\
        #     global_path.sampling_resolution, caculate_resolution(route_, np.where(route_ < np.inf, True, False).all(axis=-1))))

        ### normalize
        route[:,:2] /= self.perception_range
        route[:,2] /= np.pi
        #debug viz
        # for lane, num in zip(route, np.arange(0,route.shape[0])):
        #     lane_x, lane_y = lane[:,0], lane[:,1]
        route_x, route_y = route[:,0], route[:,1]

        return rllib.basic.Data(route=route, route_mask=route_mask)


class PerceptionVectorizedMap(object):
    invalid_value = np.inf
    dim_state = 4

    def __init__(self, config, topology_map: universe.common.TopologyMap):
        self.config = config
        self.topology_map = topology_map
        self.perception_range = config.perception_range

        # self.max_curvature = 0.1
        # self.max_resolution = 4.0
        # centerline, _= sample_array(self.topology_map.centerline, self.topology_map.centerline_mask,self.max_curvature,self.max_resolution)
        # sideline, _ = sample_array(self.topology_map.sideline, self.topology_map.sideline_mask,self.max_curvature,self.max_resolution)
        # #debug
        # print("centerline shape before sampling: {}, after sampling: {}\n".format(self.topology_map.centerline.shape, centerline.shape))
        # print("sideline shape before sampling: {}, after sampling: {}\n".format(self.topology_map.sideline.shape, sideline.shape))
        # self.topology_map = universe.common.TopologyMap(centerline,sideline)
        
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
        return rllib.basic.Data(lane=lane, lane_mask=lane_mask, bound=bound, bound_mask=bound_mask, bound_flag=bound_flag)



    def check_in_bound(self, agent: universe.common.EndToEndVehicle, bound):
        #检测严格程度[0.1, 1]
        scale = 0.6
        #设定碰撞检测范围就是30m
        range = 30.0
        state0 = agent.get_state()
        bounds = bound[..., :2]

        #取距离小于30m的bound
        dist = np.sqrt(bounds[...,0]**2 + bounds[...,1]**2)

        ### ! warning
        if np.where(dist <= np.hypot(agent.bbx_x, agent.bbx_y) *scale +0.1, 1, 0).sum() == 0:
            return np.array(False)

        valid_masks = np.where(dist < range, 1, 0)
        valid_lengths = valid_masks.sum(-1)
        max_valid_lengths = max(valid_lengths)

        sorted_index = np.expand_dims(np.argsort(-valid_masks, axis=1, kind='mergesort'), axis=2).repeat(bounds.shape[2], axis=2)
        valid_masks = np.expand_dims(valid_masks, axis=2).repeat(bounds.shape[2], axis=2)
        valid_bounds = np.where(valid_masks, bounds, np.inf)
        valid_bounds = np.take_along_axis(valid_bounds, sorted_index, axis=1)[:,:max_valid_lengths]
        valid_bounds = np.delete(valid_bounds, np.where(valid_lengths == 0)[0], axis=0).copy()
        if(valid_bounds.size == 0):
            return np.array(False)

        valid_bounds_lines, valid_masks_lines = self.trans_points_to_lines(valid_bounds)
        agent_profile_lines = np.empty((4, 4), dtype=np.float32)

        theta = 0
        
        # cu.basic.pi2pi
        x, y = agent.bbx_x * scale, agent.bbx_y * scale
        ##车长两条线
        agent_profile_lines[0,0], agent_profile_lines[0,1]= -np.cos(theta)*x - np.sin(theta)*y, -np.sin(theta)*x + np.cos(theta)*y
        agent_profile_lines[0,2], agent_profile_lines[0,3]= np.cos(theta)*x - np.sin(theta)*y, np.sin(theta)*x + np.cos(theta)*y
        agent_profile_lines[1,0], agent_profile_lines[1,1]= -np.cos(theta)*x + np.sin(theta)*y, -np.sin(theta)*x - np.cos(theta)*y 
        agent_profile_lines[1,2], agent_profile_lines[1,3]= np.cos(theta)*x + np.sin(theta)*y, np.sin(theta)*x - np.cos(theta)*y
        #车宽两条线
        agent_profile_lines[2,0], agent_profile_lines[2,1]= agent_profile_lines[0,0], agent_profile_lines[0,1]
        agent_profile_lines[2,2], agent_profile_lines[2,3]= agent_profile_lines[1,0], agent_profile_lines[1,1]
        agent_profile_lines[3,0], agent_profile_lines[3,1]= agent_profile_lines[0,2], agent_profile_lines[0,3] 
        agent_profile_lines[3,2], agent_profile_lines[3,3]= agent_profile_lines[1,2], agent_profile_lines[1,3]

        cross_point_check_data_A = np.expand_dims(agent_profile_lines, axis = 1).repeat(valid_bounds_lines.shape[1], axis=1)
        cross_point_check_data_A = np.expand_dims(cross_point_check_data_A, axis = 1).repeat(valid_bounds_lines.shape[0], axis=1)
        cross_point_check_data_B = np.expand_dims(valid_bounds_lines, axis = 0).repeat(cross_point_check_data_A.shape[0], axis=0)
        cross_point_check_data = np.concatenate([cross_point_check_data_A, cross_point_check_data_B], axis = -1)
        valid_masks_lines = np.expand_dims(valid_masks_lines, axis = 0).repeat(cross_point_check_data.shape[0], axis = 0)
        valid_masks_lines = np.expand_dims(valid_masks_lines, axis = -1).repeat(2, axis = -1)

        check_point_res = self.cross_point_check(cross_point_check_data, valid_masks_lines)
        check_point_res = check_point_res.astype(np.int).sum().astype(np.bool)
        return check_point_res


    def trans_points_to_lines(self, points: np.array):
        #vec (a,b,2) -> lines (a,b-1,4)
        lines = np.empty((points.shape[0], points.shape[1] - 1, 4), dtype = np.float32)
        lines[:,:,:2]  = points[:,:-1,:2]
        lines[:,:,2:4] = points[:,1:,:2]
        valid_masks = np.where(lines[:,:,3] == np.inf, 0, 1)
        return lines, valid_masks


    def cross_point_check(self, cross_point_check_data: np.array, cross_point_check_masks: np.array) :
        # 参考:https://segmentfault.com/a/1190000004457595?f=tt ,将每条ployline的seg的四个线段和其他polyline做相交检测
        # cross_point_check_data.shape 是 [...,8] 最后一维是两条线段
        # cross_point_check_masks.shape 是 [...,2]
        cross_point_check_data = np.where(cross_point_check_data == np.inf, np.nan, cross_point_check_data)
        vec_AC =  np.where(cross_point_check_masks,cross_point_check_data[...,4:6] - cross_point_check_data[...,0:2], np.nan)
        vec_AD =  np.where(cross_point_check_masks,cross_point_check_data[...,6:8] - cross_point_check_data[...,0:2], np.nan)
        vec_BC =  np.where(cross_point_check_masks,cross_point_check_data[...,4:6] - cross_point_check_data[...,2:4], np.nan)
        vec_BD =  np.where(cross_point_check_masks,cross_point_check_data[...,6:8] - cross_point_check_data[...,2:4], np.nan)
        vec_CA, vec_CB, vec_DA, vec_DB = vec_AC, vec_BC, vec_AD, vec_BD
        ZERO = 1e-2
        vec_product_AC_AD = np.where(cross_point_check_masks[...,0],(vec_AC[...,0]*vec_AD[...,1] - vec_AD[...,0]*vec_AC[...,1]), np.inf)
        vec_product_BC_BD = np.where(cross_point_check_masks[...,0],(vec_BC[...,0]*vec_BD[...,1] - vec_BD[...,0]*vec_BC[...,1]), np.inf)
        vec_product_CA_CB = np.where(cross_point_check_masks[...,0],(vec_CA[...,0]*vec_CB[...,1] - vec_CB[...,0]*vec_CA[...,1]), np.inf)
        vec_product_DA_DB = np.where(cross_point_check_masks[...,0],(vec_DA[...,0]*vec_DB[...,1] - vec_DB[...,0]*vec_DA[...,1]), np.inf)
        bool_res_part_1 = np.array(np.where((vec_product_AC_AD*vec_product_BC_BD)<=ZERO, 1, 0), dtype=np.bool)
        bool_res_part_2 = np.array(np.where((vec_product_CA_CB*vec_product_DA_DB)<=ZERO, 1, 0), dtype=np.bool)
        cross_point_check_res = np.logical_and(bool_res_part_1, bool_res_part_2)
        # cross_point_check_res.shape 是 [...] 最后一维是两条线段,相当于少了一维,储存的是是否相交
        return cross_point_check_res


class MapAdaptiveResolution(PerceptionVectorizedMap):
    def __init__(self, config, topology_map: universe.common.TopologyMap, sampling_resolution, sampling_curvature_up_bound):
        self.config = config
        self.topology_map = topology_map
        self.perception_range = config.perception_range

        self.max_curvature = sampling_curvature_up_bound
        self.max_resolution = sampling_resolution
        centerline, _= sample_array(self.topology_map.centerline, self.topology_map.centerline_mask,self.max_curvature,self.max_resolution)
        sideline, _ = sample_array(self.topology_map.sideline, self.topology_map.sideline_mask,self.max_curvature,self.max_resolution)
        #debug
        print("centerline shape before sampling: {}, after sampling: {}\n".format(self.topology_map.centerline.shape, centerline.shape))
        print("sideline shape before sampling: {}, after sampling: {}\n".format(self.topology_map.sideline.shape, sideline.shape))
        self.topology_map = universe.common.TopologyMap(centerline,sideline)



def sample_array(array : np.ndarray, mask : np.ndarray, max_curvature, max_resolution):
    def sample_line(line : np.ndarray, mask : np.ndarray):
            length = mask.sum()
            x, y = line[:length,0], line[:length,1]
            dx, dy = np.diff(x), np.diff(y)
            theta = np.arctan2(dy, dx)
            theta = np.append(theta, theta[-1])
            curvatures, distances = calc_curvature_with_yaw_diff(x, y, theta)
            curvatures = np.absolute(curvatures)
            sampled_line = line[:1]
            start_index = 0
            end_index = 1
            assert length >= 3 , "the line length is too short"
            while end_index < length:
                d_curvature = curvatures[start_index:end_index].sum()
                d_distances = distances[start_index:end_index].sum()

                if d_curvature > max_curvature or d_distances> max_resolution:
                    if (end_index - start_index) > 1: 
                        sampled_line = np.concatenate([sampled_line, line[end_index-1:end_index]], axis = 0)
                        start_index = end_index - 1
                    else:
                        sampled_line = np.concatenate([sampled_line, line[end_index:end_index+1]], axis = 0)
                        start_index = end_index
                        end_index += 1
                else: end_index += 1      

            sampled_line = np.concatenate([sampled_line, line[-1:]], axis = 0)

            return sampled_line
        
        # _,_,_ = caculate_resolution_2(topology_map.centerline,topology_map.centerline_mask)
    sampled_lines = []
    for line, mask in zip(array, mask):
        sampled_lines.append(sample_line(line,mask)) 

    lens = [len(line) for line in sampled_lines]
    max_len = max(lens)
    arr = np.zeros((len(sampled_lines),max_len,3),np.float32)
    mask = np.arange(max_len) < np.array(lens)[:,None]
    mask = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
    for index in np.arange(0, len(sampled_lines)):
        pad_len = max_len - sampled_lines[index].shape[0]
        pad_array = np.full((pad_len,3), np.inf)
        sampled_lines[index] = np.concatenate([sampled_lines[index],pad_array],axis=0)

    sampled_array = np.array(sampled_lines, dtype=np.float32)
    return sampled_array, mask[...,0]

def caculate_resolution(array : np.ndarray, mask : np.ndarray):
    '''array : shape is (num_lines, num_points, num_features), left align'''
    def one_line_resolution(line : np.ndarray, mask : np.ndarray):
            length = mask.sum()
            x, y = line[:length,0], line[:length,1]
            dx, dy = np.diff(x), np.diff(y)
            theta = np.arctan2(dy, dx)
            theta = np.append(theta, theta[-1])
            _, distances = calc_curvature_with_yaw_diff(x, y, theta)
            return distances
        
    distances = np.empty((0,), dtype=np.float32)
    for line, mask_ in zip(array, mask):
        distances = np.concatenate([distances, one_line_resolution(line,mask_)], axis=0)  
    distance = np.average(distances)

    return distance