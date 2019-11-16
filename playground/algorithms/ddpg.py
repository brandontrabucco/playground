"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.algorithms.algorithm import Algorithm
import tensorflow as tf


class DDPG(Algorithm):

    def __init__(
            self,
            policy,
            target_policy,
            qf,
            target_qf,
            replay_buffer,
            reward_scale=1.0,
            discount=0.99,
            observation_key="observation",
            batch_size=32,
            update_every=1,
            update_after=0,
            logger=None,
            logging_prefix="ddpg/"
    ):
        # train a policy using the deep deterministic policy gradient
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
        self.target_policy = target_policy
        self.qf = qf
        self.target_qf = target_qf

        # select into the observation dictionary
        self.observation_key = observation_key

        # control some parameters that are important for ddpg
        self.reward_scale = reward_scale
        self.discount = discount

    def update_algorithm(
            self,
            observations,
            actions,
            rewards,
            next_observations,
            terminals
    ):
        # select from the observation dictionary
        observations = observations[self.observation_key]
        next_observations = next_observations[self.observation_key]

        # build a tape to collect gradients from the policy and critics
        with tf.GradientTape(persistent=True) as tape:
            mean_actions, log_pi = self.policy.expected_value(observations)
            next_mean_actions, next_log_pi = self.target_policy.expected_value(
                next_observations)

            # build the q function target value
            inputs = tf.concat([next_observations, next_mean_actions], -1)
            target_qf_value = self.target_qf(inputs)[..., 0]
            self.record("target_qf_value", tf.reduce_mean(target_qf_value).numpy())
            qf_targets = tf.stop_gradient(
                self.reward_scale * rewards + terminals * self.discount * (
                    target_qf_value))
            self.record("qf_targets", tf.reduce_mean(qf_targets).numpy())

            # build the q function loss
            inputs = tf.concat([observations, actions], -1)
            qf_value = self.qf(inputs)[..., 0]
            self.record("qf_value", tf.reduce_mean(qf_value).numpy())
            qf_loss = tf.reduce_mean(tf.keras.losses.logcosh(qf_targets, qf_value))
            self.record("qf_loss", qf_loss.numpy())

            # build the policy loss
            inputs = tf.concat([observations, mean_actions], -1)
            policy_qf_value = self.qf(inputs)[..., 0]
            self.record("policy_qf_value", tf.reduce_mean(policy_qf_value).numpy())
            policy_loss = -tf.reduce_mean(policy_qf_value)
            self.record("policy_loss", policy_loss.numpy())

        # back prop gradients
        self.policy.apply_gradients(
            self.policy.compute_gradients(policy_loss, tape))
        self.qf.apply_gradients(
            self.qf.compute_gradients(qf_loss, tape))

        # soft update target parameters
        self.target_policy.soft_update(self.policy.get_weights())
        self.target_qf.soft_update(self.qf.get_weights())
