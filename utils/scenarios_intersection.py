import rllib
import universe


import numpy as np
import os
from typing import Dict

from universe.common import Vector
from universe.common.vehicle import Vehicle
from .scenarios_template import ScenarioRandomization


class ScenarioIntersection(universe.Scenario):
    def generate_global_paths_new(self):
        path_path = self.config.dataset_dir.path
        paths = []
        for i in range(1, 5):
            path = np.load(os.path.join(path_path, f'global_path_{i}.npy'))[:,::10,:]
            paths.append(path)
        global_paths_numpy = np.concatenate(paths, axis=0)

        self.global_paths = []
        for global_path_numpy in global_paths_numpy:
            n = np.where(global_path_numpy < np.inf, 1,0).sum(axis=0)[0]
            global_path = universe.common.GlobalPath(x=global_path_numpy[:n,0], y=global_path_numpy[:n,1])
            self.global_paths.append(global_path)

        return

    def generate_global_paths(self):
        a1 = Vector(-70, 1.7)
        a2 = Vector(-70, 5.3)
        a3 = Vector(-5.3, 70)
        a4 = Vector(-1.7, 70)
        
        b1 = Vector(1.7, 70)
        b2 = Vector(5.3, 70)
        b3 = Vector(70, 5.3)
        b4 = Vector(70, 1.7)

        c1 = Vector(70, -1.7)
        c2 = Vector(70, -5.3)
        c3 = Vector(5.3, -70)
        c4 = Vector(1.7, -70)

        d1 = Vector(-1.7, -70)
        d2 = Vector(-5.3, -70)
        d3 = Vector(-70, -5.3)
        d4 = Vector(-70, -1.7)

        self.global_paths = [
            self.topology_map.route_planning(a1, b4),
            self.topology_map.route_planning(a1, c4),
            self.topology_map.route_planning(a2, b3),
            self.topology_map.route_planning(a2, a3),

            self.topology_map.route_planning(b1, c4),
            self.topology_map.route_planning(b1, d4),
            self.topology_map.route_planning(b2, c3),
            self.topology_map.route_planning(b2, b3),
            
            self.topology_map.route_planning(c1, d4),
            self.topology_map.route_planning(c1, a4),
            self.topology_map.route_planning(c2, d3),
            self.topology_map.route_planning(c2, c3),

            self.topology_map.route_planning(d1, a4),
            self.topology_map.route_planning(d1, b4),
            self.topology_map.route_planning(d2, a3),
            self.topology_map.route_planning(d2, d3),
        ]

        return



    def generate_spawn_transforms(self):
        spawn_transforms = {}
        special_spawn_transforms = {}
        for global_path in self.global_paths:
            sts = global_path.transforms[10:10+13:4]
            special_spawn_transforms[sts[0]] = global_path
            for st in sts:
                spawn_transforms[st] = global_path
            
            ### @todo: remove
            sts_extra = global_path.transforms[45]
            spawn_transforms[sts_extra] = global_path

        ### @todo: remove future
        def shuffle_dict(a: Dict):
            import random
            key = list(a.keys())
            random.shuffle(key)
            b = {}
            for key_i in key:
                b[key_i] = a[key_i]
            return b
        spawn_transforms = shuffle_dict(spawn_transforms)
        special_spawn_transforms = shuffle_dict(special_spawn_transforms)

        _spawn_transforms = [universe.common.HashableTransform(sp) for sp in spawn_transforms.keys()]
        _spawn_transforms = [hsp.transform for hsp in list(set(_spawn_transforms))]
        self.spawn_transforms = {st: spawn_transforms[st]  for st in _spawn_transforms}

        _special_spawn_transforms = [universe.common.HashableTransform(sp) for sp in special_spawn_transforms.keys()]
        _special_spawn_transforms = [hsp.transform for hsp in list(set(_special_spawn_transforms))]
        self.special_spawn_transforms = {st: special_spawn_transforms[st] for st in _special_spawn_transforms}
        print(rllib.basic.prefix(self) + f'num spawn_transforms: {len(self.spawn_transforms)}')
        return



    def at_junction(self, vehicle: Vehicle):
        t = vehicle.get_transform()
        x, y = t.x, t.y
        flag = False
        if x >= -12.5 and x <= 12.5 and y >= -12.5 and y <= 12.5:
            flag = True
        return flag


    def finish_task(self, vehicle: Vehicle):
        state = vehicle.get_state()
        theta1 = state.theta
        theta2 = np.arctan2(self.center.y - state.y, self.center.x - state.x)
        dot = np.cos(theta1)*np.cos(theta2) + np.sin(theta1)*np.sin(theta2)
        dist = np.hypot(self.center.y - state.y, self.center.x - state.x)
        return dot * dist < -12.5





############################################################################################
#### evaluate ##############################################################################
############################################################################################



class ScenarioIntersectionEvaluate(ScenarioIntersection):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return



class ScenarioIntersectionEvaluate_assign(ScenarioIntersectionEvaluate):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[:] = round(self.env_index *0.1, 1)
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.num_vehicles, self.scenario_randomization.characters)
        return



class ScenarioIntersectionEvaluate_without_mismatch(ScenarioIntersectionEvaluate):  ### only for single-agent
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[0] = 0
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return

