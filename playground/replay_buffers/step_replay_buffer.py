"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground import nested_apply
from playground.replay_buffers.replay_buffer import ReplayBuffer
import numpy as np


class StepReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            max_num_steps=1000000
    ):
        ReplayBuffer.__init__(self)

        # parameters to control how the buffer is created and managed
        self.max_num_steps = max_num_steps

    def inflate_backend(
            self,
            x
    ):
        # create numpy arrays to store samples
        x = x if isinstance(x, np.ndarray) else np.array(x)
        return np.zeros_like(x, shape=[self.max_num_steps, *x.shape])

    def insert_backend(
            self,
            structure,
            data
    ):
        # insert samples into the numpy array
        structure[self.head, ...] = data

    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # insert a path into the replay buffer
        self.total_paths += 1
        observations = observations[:self.max_num_steps]
        actions = actions[:self.max_num_steps]
        rewards = rewards[:self.max_num_steps]

        # inflate the replay buffer if not inflated
        if any([self.observations is None, self.actions is None, self.rewards is None,
                self.terminals is None]):
            self.observations = nested_apply(self.inflate_backend, observations[0])
            self.actions = nested_apply(self.inflate_backend, actions[0])
            self.rewards = self.inflate_backend(np.squeeze(rewards[0]))
            self.terminals = self.inflate_backend(np.array([0, 0]))

        # insert all samples into the buffer
        for time_step, (o, a, r) in enumerate(zip(observations, actions, rewards)):
            nested_apply(self.insert_backend, self.observations, o)
            nested_apply(self.insert_backend, self.actions, a)
            self.insert_backend(self.rewards, np.squeeze(r))
            self.insert_backend(self.terminals, np.array([time_step, self.total_paths]))

            # increment the head and size
            self.head = (self.head + 1) % self.max_num_steps
            self.size = min(self.size + 1, self.max_num_steps)
            self.total_steps += 1

    def sample(
            self,
            batch_size
    ):
        # handle cases when we want to sample everything
        batch_size = batch_size if batch_size > 0 else self.size

        # sample transition for a hierarchy of policies
        idx = np.random.choice(
            self.size, size=batch_size, replace=(self.size < batch_size))

        def inner_sample(data):
            return data[idx, ...]

        def inner_sample_next(data):
            return data[(idx + 1) % self.size, ...]

        # sample current batch from a nested samplers agents
        observations = nested_apply(inner_sample, self.observations)
        actions = inner_sample(self.actions)
        rewards = inner_sample(self.rewards)
        next_observations = nested_apply(inner_sample_next, self.observations)

        terminals = np.equal(
            inner_sample_next(self.terminals[:, 1]),
            inner_sample(self.terminals[:, 1])).astype(np.float32)

        # return the samples in a batch
        return observations, actions, rewards, next_observations, terminals
