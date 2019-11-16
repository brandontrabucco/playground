"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from gym.spaces import Box, Dict
from gym import Env
import numpy as np


class PointmassEnv(Env):

    def __init__(
        self,
        size=2,
        order=2,
        action_scale=0.1,
    ):
        self.observation_space = Dict({
            "observation": Box(-1.0 * np.ones([size * 2]), np.ones([size * 2]))})
        self.action_space = Box(-1.0 * np.ones([size]), np.ones([size]))
        self.size = size
        self.order = order
        self.action_scale = action_scale
        self.position = np.zeros([self.size])
        self.goal = np.random.uniform(low=-1.0, high=1.0, size=[self.size])

    def reset(
        self,
        **kwargs
    ):
        self.position = np.zeros([self.size])
        self.goal = np.random.uniform(low=-1.0, high=1.0, size=[self.size])
        return {"observation": np.concatenate([self.position, self.goal], 0)}

    def step(
        self, 
        action
    ):
        clipped_action = np.clip(action, -1.0 * np.ones([self.size]), np.ones([self.size]))
        scaled_action = clipped_action * self.action_scale
        self.position = np.clip(
            self.position + scaled_action, -1.0 * np.ones([self.size]), np.ones([self.size]))
        reward = -1.0 * np.linalg.norm(self.position - self.goal, ord=self.order)
        return ({"observation": np.concatenate([self.position, self.goal], 0)},
                reward, False, {})

    def render(
        self,
        image_size=256,
        **kwargs
    ):
        image = np.zeros([image_size, image_size, 3])
        x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
        goal_radius = np.sqrt((x - (self.goal[0] + 1.0) * image_size / 2)**2 + (
            y - (self.goal[1] + 1.0) * image_size / 2)**2)
        position_radius = np.sqrt((x - (self.position[0] + 1.0) * image_size / 2)**2 + (
            y - (self.position[1] + 1.0) * image_size / 2)**2)
        image[:, :, 1] = np.ones(goal_radius.shape) / (1.0 + goal_radius / image_size * 12.0)
        image[:, :, 2] = np.ones(position_radius.shape) / (1.0 + position_radius / image_size * 12.0)
        return image
