"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.samplers.sampler import Sampler


class SequentialSampler(Sampler):

    def __init__(
            self,
            env,
            policy,
            max_path_length=1000,
            observation_key="observation"
    ):
        # parameters to control sampling from the environment.
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.observation_key = observation_key

    def set_weights(
            self,
            weights
    ):
        # set the weights for the agent in this sampler
        self.policy.set_weights(weights)

    def collect(
            self,
            min_num_steps_to_collect,
            deterministic=False,
            keep_data=True,
            render=False,
            render_kwargs=None
    ):
        # collect num_episodes amount of paths and track various things
        if render_kwargs is None:
            render_kwargs = {}

        # store data to pass to the replay buffer
        paths = []
        returns = []

        # start collecting many trajectories
        num_steps_collected = 0
        while num_steps_collected < min_num_steps_to_collect:

            # keep track of observations actions and rewards
            observations = []
            actions = []
            rewards = []

            # reset the environment at the start of each trajectory
            observation_t = self.env.reset()
            path_return = 0.0

            # unroll the episode until done or max_path_length is attained
            for time_step in range(self.max_path_length):

                # check if the environment has a goal and send it in
                num_steps_collected += 1
                inputs = observation_t[self.observation_key][None, ...]
                action_t = (self.policy.expected_value(
                    inputs) if deterministic else self.policy.sample(inputs))[0][0, ...].numpy()

                # save the observation and the actions from the agent
                if keep_data:
                    observations.append(observation_t)
                    actions.append(action_t)

                # update the environment with the atomic actions
                observation_t, reward_t, done, info = self.env.step(action_t)
                path_return += reward_t
                if keep_data:
                    rewards.append(reward_t)

                # and possibly render the updated environment (to a video)
                if render:
                    self.env.render(**render_kwargs)

                # exit if the simulation has reached a terminal state
                if done:
                    break

            # save the episode into a list to send to the replay buffer
            returns.append(path_return)
            if keep_data:
                paths.append((observations, actions, rewards))

        # return the paths and the number of steps collected so far
        return paths, returns, num_steps_collected
