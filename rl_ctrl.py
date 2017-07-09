import numpy as np


class RlController:
    def __init__(self, commander):
        self.cmd = commander

    def interact(self, action):

        reward = self.cmd.action(action)
        observation = self.cmd.get_observation()  # returns np.array

        return observation, reward


