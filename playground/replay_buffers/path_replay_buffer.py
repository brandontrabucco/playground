"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground import nested_apply
from playground.replay_buffers.replay_buffer import ReplayBuffer
import numpy as np


class PathReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            max_path_length=1000,
            max_num_paths=1000
    ):
        ReplayBuffer.__init__(self)

        # control the storage size of the replay buffer
        self.max_path_length = max_path_length
        self.max_num_paths = max_num_paths

    def inflate_backend(
            self,
            x
    ):
        # create numpy arrays to store samples
        x = x if isinstance(x, np.ndarray) else np.array(x)
        return np.zeros_like(x, shape=[self.max_num_paths, self.max_path_length, *x.shape])

    def insert_backend(
            self,
            structure,
            data
    ):
        # insert samples into the numpy array
        structure[self.head, int(self.terminals[self.head]) % self.max_path_length, ...] = data

    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # insert a path into the replay buffer
        self.total_paths += 1
        observations = observations[:self.max_path_length]
        actions = actions[:self.max_path_length]
        rewards = rewards[:self.max_path_length]

        # inflate the replay buffer if not inflated
        if any([self.observations is None, self.actions is None, self.rewards is None,
                self.terminals is None]):
            self.observations = nested_apply(self.inflate_backend, observations[0])
            self.actions = nested_apply(self.inflate_backend, actions[0])
            self.rewards = self.inflate_backend(rewards[0])
            self.terminals = np.zeros([self.max_num_paths], dtype=np.int32)

        # insert all samples into the buffer
        for time_step, (o, a, r) in enumerate(zip(observations, actions, rewards)):
            self.terminals[self.head] = time_step
            nested_apply(self.insert_backend, self.observations, o)
            nested_apply(self.insert_backend, self.actions, a)
            self.insert_backend(self.rewards, r)
            self.total_steps += 1

        # increment the head and size
        self.head = (self.head + 1) % self.max_num_paths
        self.size = min(self.size + 1, self.max_num_paths)

    def sample(
            self,
            batch_size
    ):
        # handle cases when we want to sample everything
        batch_size = batch_size if batch_size > 0 else self.size

        # determine which steps to sample from
        idx = np.random.choice(
            self.size, size=batch_size, replace=(self.size < batch_size))

        def inner_sample(data):
            return data[idx, ...]

        # sample current batch from a nested samplers agents
        observations = nested_apply(inner_sample, self.observations)
        actions = inner_sample(self.actions)
        rewards = inner_sample(self.rewards)

        # determine if the step that has been sampled is valid
        terminals = np.less_equal(
            np.arange(self.max_path_length)[None], self.terminals[idx, None]).astype(np.float32)

        # return the samples in a batch
        return observations, actions, rewards, terminals
