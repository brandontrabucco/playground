"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground import maybe_initialize_process
from abc import ABC, abstractmethod
import pickle as pkl
import tensorflow as tf


class Distribution(ABC):

    def __init__(
            self,
            model,
            tau=0.01,
            optimizer_class=tf.keras.optimizers.Adam,
            optimizer_kwargs=None,
    ):
        # wrap around a model to make it probabilistic
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.model = model
        self.tau = tau
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer_class(**optimizer_kwargs)

    def clone(
            self
    ):
        # return a copy of the model with the same weights
        return pkl.loads(pkl.dumps(self))

    def __getstate__(
            self
    ):
        # handle pickle actions so the agent can be sent between threads
        return dict(
            tau=self.tau,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            model_json=self.model.to_json(),
            model_weights=self.model.get_weights())

    def __setstate__(
            self,
            state
    ):
        # initialize tensorflow and the multiprocessing interface
        maybe_initialize_process()

        # handle pickle actions so the agent can be sent between threads
        self.model = tf.keras.models.model_from_json(state["model_json"])
        self.model.set_weights(state["model_weights"])
        self.tau = state["tau"]

    def __call__(
            self,
            *args
    ):
        # wrapper function to call the keras model
        return self.model(*args)

    def soft_update(
            self,
            weights
    ):
        # wrapper functions to transfer model state
        self.model.set_weights([
            self.tau * w1 + (1.0 - self.tau) * w2 for w1, w2 in zip(
                weights, self.model.get_weights())])

    def compute_gradients(
            self,
            loss,
            tape
    ):
        # apply the gradient update rule to this model
        return tape.gradient(loss, self.model.trainable_variables)

    def apply_gradients(
            self,
            gradients
    ):
        # apply the gradient update rule to this model
        self.optimizer.apply_gradients(zip(
            gradients, self.model.trainable_variables))

    @abstractmethod
    def get_parameters(
            self,
            *inputs
    ):
        return NotImplemented

    @abstractmethod
    def sample(
            self,
            *inputs
    ):
        return NotImplemented

    @abstractmethod
    def expected_value(
            self,
            *inputs
    ):
        return NotImplemented

    @abstractmethod
    def log_prob(
            self,
            *inputs
    ):
        return NotImplemented

    @abstractmethod
    def kl_divergence(
            self,
            *inputs
    ):
        return NotImplemented

    def prob(
            self,
            *inputs
    ):
        # compute the probability density of the inputs
        return tf.exp(self.log_prob(*inputs))

    def __setattr__(
        self,
        attr,
        value
    ):
        # pass attribute assignments to the torch model
        if not attr == "model":
            setattr(self.model, attr, value)
        else:
            self.__dict__[attr] = value

    def __getattr__(
        self,
        attr
    ):
        # pass attribute lookups to the torch model
        if not attr == "model":
            return getattr(self.model, attr)
        else:
            return self.__dict__[attr]
