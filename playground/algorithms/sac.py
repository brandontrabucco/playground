"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.algorithms.algorithm import Algorithm
import tensorflow as tf
import math


class SAC(Algorithm):

    def __init__(
            self,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            replay_buffer,
            reward_scale=1.0,
            discount=0.99,
            initial_alpha=0.1,
            alpha_optimizer=tf.keras.optimizers.Adam,
            alpha_optimizer_kwargs=None,
            target_entropy=0.0,
            observation_key="observation",
            batch_size=32,
            update_every=1,
            update_after=0,
            logger=None,
            logging_prefix="sac/"
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
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        # attributes for computing entropy tuning
        self.log_alpha = tf.Variable(math.log(abs(initial_alpha)), dtype=tf.float32)
        if alpha_optimizer_kwargs is None:
            alpha_optimizer_kwargs = {}
        self.alpha_optimizer = alpha_optimizer(
            [self.log_alpha], **alpha_optimizer_kwargs)

        # select into the observation dictionary
        self.observation_key = observation_key

        # control some parameters that are important for sac
        self.reward_scale = reward_scale
        self.discount = discount
        self.initial_alpha = initial_alpha
        self.target_entropy = target_entropy

    def update_algorithm(
            self,
            observations,
            actions,
            rewards,
            next_observations,
            terminals
    ):
        # select from the observation dictionary
        observations = tf.constant(observations[self.observation_key])
        next_observations = tf.constant(next_observations[self.observation_key])

        # build a tape to collect gradients from the policy and critics
        with tf.GradientTape(persistent=True) as tape:
            alpha = tf.exp(self.log_alpha)
            self.record("alpha", alpha.numpy())

            # sample actions from current policy
            sampled_actions, log_pi = self.policy.sample(observations)
            self.record("entropy", tf.reduce_mean(-log_pi).numpy())
            next_sampled_actions, next_log_pi = self.policy.sample(next_observations)
            self.record("next_entropy", tf.reduce_mean(-next_log_pi).numpy())

            # build q function target value
            inputs = tf.concat([next_observations, next_sampled_actions], -1)
            target_qf1_value = self.target_qf1(inputs)[..., 0]
            self.record("target_qf1_value", tf.reduce_mean(target_qf1_value).numpy())
            target_qf2_value = self.target_qf2(inputs)[..., 0]
            self.record("target_qf2_value", tf.reduce_mean(target_qf2_value).numpy())
            qf_targets = tf.stop_gradient(
                self.reward_scale * rewards + terminals * self.discount * (
                        tf.minimum(target_qf1_value,
                                   target_qf2_value) - alpha * next_log_pi))
            self.record("qf_targets", tf.reduce_mean(qf_targets).numpy())

            # build q function loss
            inputs = tf.concat([observations, actions], -1)
            qf1_value = self.qf1(inputs)[..., 0]
            self.record("qf1_value", tf.reduce_mean(qf1_value).numpy())
            qf2_value = self.qf2(inputs)[..., 0]
            self.record("qf2_value", tf.reduce_mean(qf2_value).numpy())
            qf1_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(qf_targets, qf1_value))
            self.record("qf1_loss", qf1_loss.numpy())
            qf2_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(qf_targets, qf2_value))
            self.record("qf2_loss", qf2_loss.numpy())

            # build policy loss
            inputs = tf.concat([observations, sampled_actions], -1)
            policy_qf1_value = self.qf1(inputs)[..., 0]
            self.record("policy_qf1_value", tf.reduce_mean(policy_qf1_value).numpy())
            policy_qf2_value = self.qf2(inputs)[..., 0]
            self.record("policy_qf2_value", tf.reduce_mean(policy_qf2_value).numpy())
            policy_loss = tf.reduce_mean(alpha * log_pi - tf.minimum(
                policy_qf1_value, policy_qf2_value))
            self.record("policy_loss", policy_loss.numpy())

            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(
                log_pi + self.target_entropy))
            self.record("alpha_loss", alpha_loss.numpy())

        # back prop gradients
        self.policy.apply_gradients(
            self.policy.compute_gradients(policy_loss, tape))
        self.qf1.apply_gradients(
            self.qf1.compute_gradients(qf1_loss, tape))
        self.qf2.apply_gradients(
            self.qf2.compute_gradients(qf2_loss, tape))
        self.alpha_optimizer.apply_gradients(
            zip(tape.gradient(alpha_loss, [self.log_alpha]), [self.log_alpha]))

        # soft update target parameters
        self.target_qf1.soft_update(self.qf1.get_weights())
        self.target_qf2.soft_update(self.qf2.get_weights())
