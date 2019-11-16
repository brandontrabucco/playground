"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(
            self,
            replay_buffer,
            batch_size=32,
            update_every=1,
            update_after=0,
            logger=None,
            logging_prefix="algorithm/"
    ):
        # the replay buffer for computing samples fo the algorithm
        self.replay_buffer = replay_buffer
        
        # batch size for samples form replay buffer
        self.batch_size = batch_size

        # specify the number of episodes to collect
        self.update_every = update_every
        self.update_after = update_after

        # logging
        self.logger = logger
        self.logging_prefix = logging_prefix

        # necessary for update_every and update_after
        self.last_update_iteration = -1

    def record(
            self,
            key,
            value
    ):
        # record a value using the monitor
        if self.logger is not None:
            self.logger.record(self.logging_prefix + key, value)

    @abstractmethod
    def update_algorithm(
            self,
            *args
    ):
        return NotImplemented

    def fit(
            self,
            iteration
    ):
        # only train on certain steps
        if (iteration >= self.update_after) and (
                iteration -
                self.last_update_iteration >= self.update_every):
            self.last_update_iteration = iteration

            # get a batch of data from the replay buffer
            batch_of_data = self.replay_buffer.sample(
                self.batch_size)

            # samples are pulled from the replay buffer on the fly
            self.update_algorithm(*batch_of_data)
