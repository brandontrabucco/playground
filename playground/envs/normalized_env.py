"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground import nested_apply
from playground.envs.proxy_env import ProxyEnv
from gym.spaces import Box, Dict, Discrete
import numpy as np


def create_space(space):
    return Box(-np.ones(space.shape), np.ones(space.shape))


def denormalize(data, space):
    lower_bound = space.low
    upper_bound = space.high

    # check if the data boundaries are infinity
    skip_normalization = np.logical_or(
        np.logical_or(np.isinf(lower_bound), np.less_equal(lower_bound, -1e9)),
        np.logical_or(np.isinf(upper_bound), np.greater_equal(upper_bound, 1e9)))

    if np.any(skip_normalization):
        # the bounds are invalid so de normalization would overflow
        return data

    else:
        # proceed with de normalization
        return np.clip(
            lower_bound + (data + 1.0) * 0.5 * (
                upper_bound - lower_bound), lower_bound, upper_bound)


def normalize(data, space):
    lower_bound = space.low
    upper_bound = space.high

    # check if the data boundaries are infinity
    skip_normalization = np.logical_or(
        np.logical_or(np.isinf(lower_bound), np.less_equal(lower_bound, -1e9)),
        np.logical_or(np.isinf(upper_bound), np.greater_equal(upper_bound, 1e9)))

    if np.any(skip_normalization):
        # the bounds are invalid so normalization would overflow
        return data

    else:
        # proceed with normalization
        return np.clip(
            (data - lower_bound) * 2.0 / (
                upper_bound - lower_bound) - 1.0, -1.0, 1.0)


class NormalizedEnv(ProxyEnv):

    def __init__(
        self, 
        *args,
        **kwargs
    ):
        # normalize the action and observation space to -1 and 1
        ProxyEnv.__init__(self, *args, **kwargs)
        self.original_observation_space = self.observation_space.spaces
        self.original_action_space = self.action_space
        self.observation_space = Dict(
            nested_apply(
                create_space, self.original_observation_space))

        if not isinstance(self.original_action_space, Discrete):
            self.action_space = create_space(self.original_action_space)
        else:
            self.action_space = self.original_action_space

    def reset(
        self,
        **kwargs
    ):
        observation = ProxyEnv.reset(self, **kwargs)
        observation = nested_apply(
            normalize, observation, self.original_observation_space)
        observation = nested_apply(
            lambda x: x.astype(np.float32), observation)
        return observation

    def step(
        self, 
        action
    ):
        if not isinstance(self.original_action_space, Discrete):
            action = denormalize(
                action, self.original_action_space)
        observation, reward, done, info = ProxyEnv.step(self, action)
        observation = nested_apply(
            normalize, observation, self.original_observation_space)
        observation = nested_apply(
            lambda x: x.astype(np.float32), observation)
        return observation, reward, done, info
