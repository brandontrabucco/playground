"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.algorithms.algorithm import Algorithm
from playground import discounted_sum
import tensorflow as tf


class PolicyGradient(Algorithm):

    def __init__(
            self,
            policy,
            replay_buffer,
            reward_scale=1.0,
            discount=0.99,
            observation_key="observation",
            batch_size=32,
            update_every=1,
            update_after=0,
            logger=None,
            logging_prefix="policy_gradient/"
    ):
        # train a policy using the vanilla policy gradient
        Algorithm.__init__(
            self,
            replay_buffer,
            batch_size=batch_size,
            update_every=update_every,
            update_after=update_after,
            logger=logger,
            logging_prefix=logging_prefix)

        # the policy is a probabilistic neural network
        self.policy = policy

        # select into the observation dictionary
        self.observation_key = observation_key

        # control the scale and decay of the reward
        self.reward_scale = reward_scale
        self.discount = discount

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        # select elements from the observation dictionary
        observations = observations[self.observation_key]

        # update the policy gradient algorithm
        with tf.GradientTape() as tape:

            # compute advantages using the sampled rewards
            discounted_returns = discounted_sum(rewards, self.discount)
            self.record("discounted_returns", tf.reduce_mean(discounted_returns))
            advantages = discounted_returns - tf.reduce_mean(discounted_returns)
            self.record("advantages", tf.reduce_mean(advantages))

            # compute the surrogate policy loss
            policy_log_prob = self.policy.log_prob(actions, observations)
            self.record("policy_log_prob", tf.reduce_mean(policy_log_prob))
            policy_loss = -tf.reduce_mean(policy_log_prob * advantages)
            self.record("policy_loss", policy_loss)

        # back prop gradients into the policy
        self.policy.apply_gradients(
            self.policy.compute_gradients(policy_loss, tape))
