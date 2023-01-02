import rllib
import universe

import numpy as np
import random



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







class ScenarioRandomizationWithoutMismatch(ScenarioRandomization):
    def get_characters(self):
        characters = np.random.uniform(0,1, size=self.num_vehicles)
        characters[0] = 0.0
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)

class ScenarioRandomizationFixOtherSvo(ScenarioRandomization):
    def get_characters(self):
        characters = np.full(self.num_vehicles.size(),np.random.uniform(0,1, size=1))
        breakpoint()
        characters[0] = 0.0
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)

class ScenarioRandomizationWithoutMismatchDisSvo(ScenarioRandomization):
    def get_characters(self):
        characters = np.random.uniform(0,1, size=self.num_vehicles)
        #CHANGE at 2022/11/12 xue
        characters[1:] = np.round(characters[1:], 1)
        characters[0] = 0
        # print(rllib.basic.prefix(self) + 'characters: ', characters)
        return characters.astype(np.float32)


