import rllib
import universe


import numpy as np
import os
from typing import Dict
import random

from .scenarios_template import ScenarioRandomization, ScenarioRandomization_fix_svo



class ScenarioBottleneck(universe.Scenario):
    def generate_global_paths(self):
        path_path = self.config.dataset_dir.path
        global_paths_numpy = np.concatenate([
            np.load(os.path.join(path_path, 'bottom_all.npy')),
            np.load(os.path.join(path_path, 'upper_all.npy')),
        ], axis=0)

        self.global_paths = []
        for global_path_numpy in global_paths_numpy:
            global_path = universe.common.GlobalPath(x=global_path_numpy[:,0], y=global_path_numpy[:,1])
            self.global_paths.append(global_path)

        return


    def generate_spawn_transforms(self):
        spawn_transforms = {}
        special_spawn_transforms = {}
        for global_path in self.global_paths:
            sts = global_path.transforms[10:10+13:4]
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



class ScenarioBottleneckEvaluate(ScenarioBottleneck):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return




class ScenarioBottleneckEvaluate_assign(ScenarioBottleneckEvaluate):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[:] = round(self.env_index *0.1, 1)
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return





class ScenarioBottleneckEvaluate_fix_others(ScenarioBottleneckEvaluate):
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.config.randomization_index}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[0] = self.step_reset /10
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return

class ScenarioBottleneckEvaluate_fix_our_others(ScenarioBottleneckEvaluate):
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





class ScenarioBottleneckEvaluate_without_mismatch(ScenarioBottleneckEvaluate):  ### only for single-agent
    def generate_scenario_randomization(self):
        scenario_randomization_cls = self.config.get('scenario_randomization_cls', ScenarioRandomization)
        dir_path = os.path.join(self.config.dataset_dir.map, f'../scenario_offline/{self.config.scenario_name}')
        file_path = os.path.join(dir_path, f'{self.step_reset}.txt')
        self.scenario_randomization = scenario_randomization_cls.load(file_path)
        self.scenario_randomization.characters[0] = 0.7
        print(rllib.basic.prefix(self) + f'characters {self.step_reset}: ', self.scenario_randomization.characters)
        return


