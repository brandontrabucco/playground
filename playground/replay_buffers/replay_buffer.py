"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class ReplayBuffer(ABC):

    def __init__(
            self
    ):
        # storage agents for the samples collected
        self.observations = None
        self.actions = None
        self.rewards = None
        self.terminals = None

        # parameters to indicate the size of the buffer
        self.head = 0
        self.size = 0
        self.total_steps = 0
        self.total_paths = 0

    def empty(
            self
    ):
        # empties the replay buffer of its elements
        self.head = 0
        self.size = 0

    def get_total_paths(
            self
    ):
        # return the total number of episodes collected
        return self.total_paths

    def get_total_steps(
            self
    ):
        # return the total number of transitions collected
        return self.total_steps

    def to_dict(
            self,
    ):
        # save the replay buffer to a dictionary
        return dict(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            size=self.size,
            head=self.head,
            total_steps=self.total_steps,
            total_paths=self.total_paths)

    def from_dict(
            self,
            state
    ):
        # load the replay buffer from a dictionary
        self.observations = state["observations"]
        self.actions = state["actions"]
        self.rewards = state["rewards"]
        self.terminals = state["terminals"]

        self.size = state["size"]
        self.head = state["head"]
        self.total_steps = state["total_steps"]
        self.total_paths = state["total_paths"]

    @abstractmethod
    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        return NotImplemented

    @abstractmethod
    def sample(
            self,
            batch_size
    ):
        return NotImplemented
