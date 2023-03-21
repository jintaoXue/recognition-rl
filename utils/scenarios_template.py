from matplotlib.pyplot import axis
import rllib
import universe

import numpy as np
import random

#roundabout class
from universe.common.geo import Vector, Transform
from universe.common.actor import ActorBoundingBox, check_collision
from universe.common.global_path import GlobalPath
import copy
from typing import List, Dict

class ScenarioRandomization(universe.ScenarioRandomization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.characters = self.get_characters()

    def get_characters(self):
        characters = np.random.uniform(0,1, size=self.num_vehicles)
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)

    def __getitem__(self, vi):
        return super().__getitem__(vi) + rllib.basic.BaseData(character=self.characters[vi])



class ScenarioRandomizationRoundabout(ScenarioRandomization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __init__(self, spawn_transforms: Dict[Transform, GlobalPath], bbx_vectors: List[Vector], num_vehicles):
        self._spawn_transforms = spawn_transforms
        self.num_vehicles = num_vehicles

        if num_vehicles > len(spawn_transforms):
            msg = 'requested {} vehicles, but could only find {} spawn points'.format(num_vehicles, len(self._spawn_transforms))
            print(rllib.basic.prefix(self) + 'warning: {}'.format(msg))
        
        self.spawn_transforms: List[Transform] = np.random.choice(list(self._spawn_transforms.keys()), size=num_vehicles, replace=False)
        self.bbx_vectors = np.random.choice(bbx_vectors, size=num_vehicles, replace=False)
        
        bbxs = np.array([ActorBoundingBox(t, bbx.x, bbx.y) for t, bbx in zip(self.spawn_transforms, self.bbx_vectors)])
        valid_flag = ~check_collision(bbxs)
        self.spawn_transforms = self.spawn_transforms[valid_flag]
        self.global_paths = np.array([copy.deepcopy(self._spawn_transforms[sp]) for sp in self.spawn_transforms])
        self.num_vehicles = len(self.global_paths)

        self.characters = self.get_characters()
        return




class ScenarioRandomization_fix_svo(universe.ScenarioRandomization):
    def __init__(self, svo,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.svo = 0.0
        self.characters = self.get_characters()

    def get_characters(self):
        characters = np.array(self.num_vehicles*[self.svo])
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)

    def __getitem__(self, vi):
        return super().__getitem__(vi) + rllib.basic.BaseData(character=self.characters[vi])

class ScenarioRandomization_share_character(ScenarioRandomization):
    def get_characters(self):
        character = random.uniform(0, 1)
        characters = [character] * self.num_vehicles
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters

#diverse controll policy for background agent
class ScenarioRandomizationDivese(universe.ScenarioRandomization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.controll_types = self.get_control_type()
        self.characters = self.get_characters()

    def get_characters(self):
        characters = np.random.uniform(0,1, size=self.num_vehicles)
        # characters = 
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)

    def get_control_type(self):
        #0是ego policy, 1是flow, 2是robust
        control_types = np.random.randint(1,3, size=self.num_vehicles)
        control_types = np.sort(control_types,axis=0)
        control_types[0] = 0
        return control_types.astype(np.int32)

    def __getitem__(self, vi):
        return super().__getitem__(vi) + \
            rllib.basic.BaseData(character=self.characters[vi], control_type = self.controll_types[vi])



class ScenarioRandomizationWithoutMismatch(ScenarioRandomization):
    def get_characters(self):
        characters = np.random.uniform(0,1, size=self.num_vehicles)
        characters[0] = 0.0
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)

class ScenarioRandomizationFixOtherSvo(ScenarioRandomization):
    def get_characters(self):
        # characters = np.random.uniform(0,1, size=self.num_vehicles)
        # characters[1:] = characters[1]
        # characters[0] = 0.0
        # # print(rllib.basic.prefix(self) + 'characters: ', characters)
        # return characters.astype(np.float32)
        characters = np.full(self.num_vehicles, np.random.uniform(0,1, size=1),dtype=np.float32)
        characters[0] = 0.0
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters

class ScenarioRandomizationWithoutMismatchDisSvo(ScenarioRandomization):
    def get_characters(self):
        characters = np.random.uniform(0,1, size=self.num_vehicles)
        #CHANGE at 2022/11/12 xue
        characters[1:] = np.round(characters[1:], 1)
        characters[0] = 0
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)


