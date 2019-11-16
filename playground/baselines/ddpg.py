"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground import maybe_initialize_process
from playground.envs.normalized_env import NormalizedEnv
from playground.distributions.gaussian import Gaussian
from playground.networks import dense
from playground.replay_buffers.step_replay_buffer import StepReplayBuffer
from playground.loggers.tensorboard_logger import TensorboardLogger
from playground.samplers.parallel_sampler import ParallelSampler
from playground.algorithms.ddpg import DDPG
import numpy as np


ddpg_variant = dict(
    max_num_steps=1000000,
    logging_dir="./",
    hidden_size=400,
    num_hidden_layers=2,
    reward_scale=1.0,
    discount=0.99,
    policy_learning_rate=0.0003,
    qf_learning_rate=0.0003,
    tau=0.005,
    exploration_noise_std=0.1,
    batch_size=256,
    max_path_length=1000,
    num_workers=2,
    num_warm_up_steps=10000,
    num_steps_per_epoch=1000,
    num_steps_per_eval=10000,
    num_epochs_per_eval=1,
    num_epochs=10000)


def ddpg(
        variant,
        env_class,
        env_kwargs=None,
        observation_key="observation",
):
    # initialize tensorflow and the multiprocessing interface
    maybe_initialize_process()

    # run an experiment with multiple agents
    if env_kwargs is None:
        env_kwargs = {}

    # initialize the environment to track the cardinality of actions
    env = NormalizedEnv(env_class, **env_kwargs)
    action_dim = env.action_space.low.size
    observation_dim = env.observation_space.spaces[
        observation_key].low.size

    # create a replay buffer to store data
    replay_buffer = StepReplayBuffer(
        max_num_steps=variant["max_num_steps"])

    # create a logging instance
    logger = TensorboardLogger(
        replay_buffer, variant["logging_dir"])

    # create policies for each level in the hierarchy
    policy = Gaussian(
        dense(
            observation_dim,
            action_dim,
            hidden_size=variant["hidden_size"],
            num_hidden_layers=variant["num_hidden_layers"],
            output_activation="tanh"),
        optimizer_kwargs=dict(learning_rate=variant["policy_learning_rate"]),
        tau=variant["tau"],
        std=variant["exploration_noise_std"])
    target_policy = policy.clone()

    # create critics for each level in the hierarchy
    qf = Gaussian(
        dense(
            observation_dim + action_dim,
            1,
            hidden_size=variant["hidden_size"],
            num_hidden_layers=variant["num_hidden_layers"]),
        optimizer_kwargs=dict(learning_rate=variant["qf_learning_rate"]),
        tau=variant["tau"],
        std=1.0)
    target_qf = qf.clone()

    # train the agent using soft actor critic
    algorithm = DDPG(
        policy,
        target_policy,
        qf,
        target_qf,
        replay_buffer,
        reward_scale=variant["reward_scale"],
        discount=variant["discount"],
        observation_key=observation_key,
        batch_size=variant["batch_size"],
        logger=logger,
        logging_prefix="ddpg/")

    # make a sampler to collect data to warm up the hierarchy
    sampler = ParallelSampler(
        env,
        policy,
        max_path_length=variant["max_path_length"],
        num_workers=variant["num_workers"])

    # collect more training samples
    sampler.set_weights(policy.get_weights())
    paths, returns, num_steps = sampler.collect(
        variant["num_warm_up_steps"],
        deterministic=False,
        keep_data=True,
        workers_to_use=variant["num_workers"])

    # insert the samples into the replay buffer
    for o, a, r in paths:
        replay_buffer.insert_path(o, a, r)

    #  train for a specified number of iterations
    training_steps = 0
    for iteration in range(variant["num_epochs"]):

        if iteration % variant["num_epochs_per_eval"] == 0:

            # evaluate the policy at this step
            sampler.set_weights(policy.get_weights())
            paths, eval_returns, num_steps = sampler.collect(
                variant["num_steps_per_eval"],
                deterministic=True,
                keep_data=False,
                workers_to_use=variant["num_workers"])
            logger.record("eval_mean_return", np.mean(eval_returns))

        # collect more training samples
        sampler.set_weights(policy.get_weights())
        paths, train_returns, num_steps = sampler.collect(
            variant["num_steps_per_epoch"],
            deterministic=False,
            keep_data=True,
            workers_to_use=1)
        logger.record("train_mean_return", np.mean(train_returns))

        # insert the samples into the replay buffer
        for o, a, r in paths:
            replay_buffer.insert_path(o, a, r)

        # train once each for the number of steps collected
        for i in range(num_steps):
            algorithm.fit(training_steps)
            training_steps += 1
