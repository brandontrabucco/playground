"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.algorithms.algorithm import Algorithm
from playground import discounted_sum
import tensorflow as tf


class PPO(Algorithm):

    def __init__(
            self,
            policy,
            old_policy,
            vf,
            replay_buffer,
            reward_scale=1.0,
            discount=0.99,
            epsilon=0.2,
            lamb=0.95,
            off_policy_updates=1,
            critic_updates=1,
            observation_key="observation",
            batch_size=32,
            update_every=1,
            update_after=0,
            logger=None,
            logging_prefix="ppo/"
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

        # each neural network is probabilistic
        self.policy = policy
        self.old_policy = old_policy
        self.vf = vf

        # select into the observation dictionary
        self.observation_key = observation_key

        # control some parameters that are important for ppo
        self.reward_scale = reward_scale
        self.discount = discount
        self.epsilon = epsilon
        self.lamb = lamb
        self.off_policy_updates = off_policy_updates
        self.critic_updates = critic_updates

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        # select elements from the observation dictionary sampled
        observations = observations[self.observation_key]

        # train the value function using the discounted return
        for i in range(self.critic_updates):
            with tf.GradientTape() as tape:

                # compute the value function loss
                discounted_returns = discounted_sum(rewards, self.discount)
                self.record("discounted_returns", tf.reduce_mean(discounted_returns).numpy())
                values = self.vf(observations)[..., 0]
                self.record("values", tf.reduce_mean(values).numpy())
                vf_loss = tf.losses.mean_squared_error(discounted_returns, values)
                self.record("vf_loss", tf.reduce_mean(vf_loss).numpy())

            # back prop into the value function
            self.vf.apply_gradients(
                self.vf.compute_gradients(vf_loss, tape))

        # compute generalized advantages
        delta_v = (rewards - values +
                   self.discount * tf.pad(values, [[0, 0], [0, 1]])[:, 1:])
        self.record("delta_v", tf.reduce_mean(delta_v).numpy())
        advantages = discounted_sum(delta_v, self.discount * self.lamb)
        self.record("advantages", tf.reduce_mean(advantages).numpy())

        # train the policy using generalized advantage estimation
        self.old_policy.set_weights(self.policy.get_weights())
        for i in range(self.off_policy_updates):
            with tf.GradientTape() as tape:

                # compute the importance sampling policy ratio
                policy_ratio = tf.exp(self.policy.log_prob(actions, observations) -
                                      self.old_policy.log_prob(actions, observations))
                self.record("policy_ratio", tf.reduce_mean(policy_ratio).numpy())

                # compute the clipped surrogate loss function
                policy_loss = -tf.reduce_mean(
                    tf.minimum(
                        advantages * policy_ratio,
                        advantages * tf.clip_by_value(
                            policy_ratio, 1 - self.epsilon, 1 + self.epsilon)))
                self.record("policy_loss", tf.reduce_mean(policy_loss).numpy())

            # back prop into the policy
            self.policy.apply_gradients(
                self.policy.compute_gradients(policy_loss, tape))
