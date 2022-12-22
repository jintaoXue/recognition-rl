from ssl import cert_time_to_seconds
from turtle import distance
from matplotlib.pyplot import axis
import rllib

import numpy as np
import networkx as nx
import time
from torch import float32

from universe.common.color import ColorLib
from universe.common.geo import Vector, State
from universe.common.global_path import GlobalPath
from universe.common.topology_map import transform_points, TopologyMap
from universe.common.global_path import calc_curvature_with_yaw_diff

class TopologyMapSampled(TopologyMap):
    # sampling_resolution = 4.1
    '''
    centerline resolution before sampling: 2.0399537086486816, after sampling: 4.050001559985045
    sideline resolution before sampling: 2.0572986602783203, after sampling: 2.0572986602783203
    '''


    sampling_resolution = 4.2
    '''
    centerline resolution before sampling: 2.0399537086486816, after sampling: 4.050001559985045
    sideline resolution before sampling: 2.0572986602783203, after sampling: 4.09574217498061
    '''

    # sampling_resolution = 6.2
    '''
    centerline resolution before sampling: 2.0399537086486816, after sampling: 6.019723641659845
    sideline resolution before sampling: 2.0572986602783203, after sampling: 6.074268053686074

    '''

    # sampling_resolution = 11
    '''
    centerline resolution before sampling: 2.0399537086486816, after sampling: 9.682209968566895
    sideline resolution before sampling: 2.0572986602783203, after sampling: 9.730185508728027

    '''

    def __init__(self, centerline, sideline):
        # debug
        center_resolution_before = caculate_resolution(centerline,np.where(centerline < np.inf, True, False).all(axis=-1))
        side_resolution_before = caculate_resolution(sideline,np.where(sideline < np.inf, True, False).all(axis=-1))

        centerline = down_sample(centerline, np.where(centerline < np.inf, True, False).all(axis=-1), self.sampling_resolution)
        sideline = down_sample(sideline, np.where(sideline < np.inf, True, False).all(axis=-1), self.sampling_resolution)

        # debug
        print("centerline resolution before sampling: {}, after sampling: {}".format(\
            center_resolution_before , caculate_resolution(centerline,np.where(centerline < np.inf, True, False).all(axis=-1))))
        print("sideline resolution before sampling: {}, after sampling: {}\n".format(\
            side_resolution_before, caculate_resolution(sideline,np.where(sideline < np.inf, True, False).all(axis=-1))))

        self.centerline = centerline
        self.centerline_mask = np.where(centerline < np.inf, True, False).all(axis=-1)
        self.centerline_length = self.centerline_mask.astype(np.int64).sum(axis=1)

        self.sideline = sideline
        self.sideline_mask = np.where(sideline < np.inf, True, False).all(axis=-1)
        self.sideline_length = self.sideline_mask.astype(np.int64).sum(axis=1)
        
        dx = np.diff(np.where(self.centerline_mask, self.centerline[...,0], 0.0))
        dy = np.diff(np.where(self.centerline_mask, self.centerline[...,1], 0.0))
        theta = np.arctan2(dy, dx)
        self.centerline_theta = np.where(self.centerline_mask, np.concatenate([theta, theta[:,[-1]]], axis=1), np.inf)
        for i, cl in enumerate(self.centerline_length):
            self.centerline_theta[i,cl-1] = self.centerline_theta[i,cl-2]

        self.graph = nx.DiGraph()

        t1 = time.time()
        self.build_graph()
        t2 = time.time()
        print(rllib.basic.prefix(self) + 'build graph time: ', t2-t1)


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

def down_sample(array : np.ndarray, mask : np.ndarray, sampling_resolution):     
    '''array : shape is (num_lines, num_points, num_features), left align'''
    def sample_one_line(line : np.ndarray , mask, interval, sampled_length):

        len = mask.sum(-1)
        line = line[:len]
        if (len - 1) % interval == 0:
            '''the end point is included'''
            line = line[0::interval]
        else:
            '''when the end point is not included'''
            line = np.concatenate([line[0:-1:interval,:], line[-1:,:]], axis= 0)
        
        pad_line = np.full((sampled_length - line.shape[0], line.shape[1]), np.inf, dtype=np.float32)
        line = np.concatenate([line, pad_line], axis=0)
        return line
    pre_resolution = caculate_resolution(array, mask)
    mask.astype(int)
    i = np.clip(int(sampling_resolution / pre_resolution), 1, None)
    if i == 1 : return array
    #sampled_length depends on whether we can contain the end point
    sampled_length =  int((mask.shape[1] - 1)/i + 1 if (mask.shape[1]-1) % i == 0 else (mask.shape[1] - 1)/i + 2)
    sampled_array = np.empty((0,sampled_length,array.shape[2]), np.float32)

    for line, mask_ in zip(array, mask): 
        line = sample_one_line(line, mask_, i, sampled_length)
        line = np.expand_dims(line, axis=0)
        sampled_array = np.concatenate([sampled_array, line], axis=0)
    return sampled_array

    
