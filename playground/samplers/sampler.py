"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Sampler(ABC):

    @abstractmethod
    def set_weights(
            self,
            weights
    ):
        # set the weights for the agent in this sampler
        return NotImplemented

    @abstractmethod
    def collect(
            self,
            min_num_steps_to_collect,
            deterministic=False,
            keep_data=False,
            render=False,
            render_kwargs=None
    ):
        # collect num_episodes amount of paths and track various things
        return NotImplemented
