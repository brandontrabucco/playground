"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from playground.loggers.logger import Logger
import tensorflow as tf


class TensorboardInterface(Logger):

    def __init__(
        self,
        replay_buffer,
        logging_dir,
    ):
        # create a separate tensor board logging thread
        self.replay_buffer = replay_buffer
        self.logging_dir = logging_dir

        # create the tensor board logging file to save training data
        tf.io.gfile.makedirs(logging_dir)
        self.writer = tf.summary.create_file_writer(logging_dir)

    def record(
        self,
        key,
        value,
    ):
        # get the current number of samples collected
        tf.summary.experimental.set_step(self.replay_buffer.get_total_steps())
        with self.writer.as_default():

            # generate a plot and write the plot to tensor board
            if len(tf.shape(value)) == 1:
                pass

            # generate several plots and write the plot to tensor board
            elif len(tf.shape(value)) == 2:
                pass

            # write a single image to tensor board
            elif len(tf.shape(value)) == 3:
                tf.summary.image(key, tf.expand_dims(value, 0) * 0.5 + 0.5)

            # write several images to tensor board
            elif len(tf.shape(value)) == 4:
                tf.summary.image(key, value * 0.5 + 0.5)

            # otherwise, assume the tensor is still a scalar
            else:
                tf.summary.scalar(key, value)


class TensorboardLogger(Logger):

    def __init__(
        self,
        replay_buffer,
        logging_dir,
    ):
        # create a separate tensor board logging thread
        self.interface = TensorboardInterface(replay_buffer, logging_dir)

    def record(
        self,
        key,
        value,
    ):
        # get the current number of samples collected
        self.interface.record(key, value)
