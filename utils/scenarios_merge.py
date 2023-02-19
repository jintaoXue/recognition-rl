import rllib
import universe
from universe.common import Vector

import numpy as np
import os
from typing import Dict
import random

from .scenarios_template import ScenarioRandomization, ScenarioRandomization_fix_svo



class ScenarioMerge(universe.Scenario):
    def generate_global_paths(self):
        self.global_paths = [
            self.topology_map.route_planning(Vector(159.4, -118.4), Vector(191.2, -22)),
            self.topology_map.route_planning(Vector(160.5, -121.6), Vector(194.6, -22)),

            self.topology_map.route_planning(Vector(191.2, -138.0), Vector(191.2, -22)),
            self.topology_map.route_planning(Vector(194.6, -138.0), Vector(194.6, -22)),
            self.topology_map.route_planning(Vector(198.1, -138.0), Vector(198.1, -22)),
        ]
        return


    def generate_spawn_transforms(self):
        spawn_transforms = {}
        special_spawn_transforms = {}
        for global_path in self.global_paths:
            sts = global_path.transforms[0:0+13:4]
            special_spawn_transforms[sts[0]] = global_path
            for st in sts:
                spawn_transforms[st] = global_path

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
        self.spawn_transforms = {st: spawn_transforms[st] for st in _spawn_transforms}

        _special_spawn_transforms = [universe.common.HashableTransform(sp) for sp in special_spawn_transforms.keys()]
        _special_spawn_transforms = [hsp.transform for hsp in list(set(_special_spawn_transforms))]
        self.special_spawn_transforms = {st: special_spawn_transforms[st] for st in _special_spawn_transforms}
        return






############################################################################################
#### evaluate ##############################################################################
############################################################################################



class ScenarioMergeEvaluate(ScenarioMerge):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return



class ScenarioMergeEvaluate_assign(ScenarioMergeEvaluate):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[:] = round(self.env_index *0.1, 1)
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return


class ScenarioMergeEvaluate_without_mismatch(ScenarioMergeEvaluate):  ### only for single-agent
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[0] = 0
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return

class ScenarioBottleneckEvaluate_fix_our_others(ScenarioMergeEvaluate):
    def reset(self, ego_svo, other_svo, step_reset, sa=True):
        self.step_reset = step_reset
        self.num_vehicles = random.randint(self.num_vehicles_min, self.num_vehicles_max)
        if sa:
            self.num_agents = 1
        else:
            self.num_agents = self.num_vehicles
        self.generate_scenario_randomization(ego_svo, other_svo)
        if self.num_vehicles < self.scenario_randomization.num_vehicles:
            self.scenario_randomization.num_vehicles = self.num_vehicles
        return
    def generate_scenario_randomization(self, ego_svo, other_svo):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization_fix_svo)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[0] = ego_svo
        self.scenario_randomization.characters[1:] = other_svo
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return




class ScenarioMergeEvaluate_assign_case(ScenarioMergeEvaluate):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.config.randomization_index}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        # self.scenario_randomization.characters[0] = self.step_reset /10
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return