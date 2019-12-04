"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.distributions.distribution import Distribution
import tensorflow as tf
import math


class Gaussian(Distribution):

    def __init__(
            self,
            model,
            std=1.0,
            tau=0.01,
            optimizer_class=tf.keras.optimizers.Adam,
            optimizer_kwargs=None,
    ):
        # create a gaussian distribution with fixed or learned standard deviation
        Distribution.__init__(
            self,
            model,
            tau=tau,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs)
        self.std = std

    def __getstate__(
            self
    ):
        # handle pickle actions so the agent can be sent between threads
        state = Distribution.__getstate__(self)
        return dict(std=self.std, **state)

    def __setstate__(
            self,
            state
    ):
        # handle pickle actions so the agent can be sent between threads
        Distribution.__setstate__(self, state)
        self.std = state["std"]

    def get_parameters(
            self,
            *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        x = self.model(tf.concat(inputs, (-1)))
        if self.std is None:
            return tf.split(x, 2, axis=(-1))
        else:
            return x, tf.math.log(tf.fill(tf.shape(x), self.std))

    def sample(
            self,
            *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        mean, log_std = self.get_parameters(*inputs)
        std = tf.math.exp(log_std)

        # re parameterized sample from the distribution
        gaussian_samples = mean + tf.random.normal(tf.shape(mean)) * std

        # compute the log probability density of the samples
        return gaussian_samples, tf.reduce_sum(
            - 0.5 * ((gaussian_samples - mean) / std) ** 2
            - log_std
            - 0.5 * tf.math.log(2 * math.pi), axis=(-1))

    def expected_value(
            self,
            *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        mean, log_std = self.get_parameters(*inputs)

        # compute the log probability density of the mean
        return mean, tf.reduce_sum(
            - log_std
            - 0.5 * tf.math.log(2 * math.pi), axis=(-1))

    def log_prob(
            self,
            gaussian_samples,
            *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        mean, log_std = self.get_parameters(*inputs)
        std = tf.math.exp(log_std)

        # compute the log probability density of the samples
        return tf.reduce_sum(
            - 0.5 * ((gaussian_samples - mean) / std) ** 2
            - log_std
            - 0.5 * tf.math.log(2 * math.pi), axis=(-1))

    def kl_divergence(
            self,
            other,
            *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        mean, log_std = self.get_parameters(*inputs)
        std = tf.math.exp(log_std)

        # get the mean and the log standard deviation of the other distribution
        other_mean, other_log_std = other.get_parameters(*inputs)
        other_std = tf.math.exp(other_log_std)

        # compute the kl divergence between the distributions
        return tf.reduce_sum(
            - log_std
            + other_log_std
            - 0.5
            + 0.5 * (std / other_std) ** 2
            + 0.5 * ((other_mean - mean) / other_std) ** 2, axis=(-1))
