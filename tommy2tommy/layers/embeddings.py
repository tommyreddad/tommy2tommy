"""Embeddings.

This module contains embedding layers relevant to transformer models,
including positional encoding.

Todo:
    * Implement alternatives to "radial" positional encoding.

"""

import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """This layer adds a positional encoding to vectors in an input
    sequence.

    Args:
        max_length (int, optional): the maximum length of input sequences.
        **kwargs (dict): named arguments of the parent class.

    """

    def __init__(self, max_length=2048, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self._max_length = max_length

    def build(self, inputs_shape):
        """Build the positional encoding from the fixed input shapes.

        Args:
            inputs_shape: the shape of input Tensors.

        """
        position = tf.cast(self._max_length, tf.float32)
        d_model = tf.cast(inputs_shape[-1], tf.float32)

        # Get the angles for the positional encoding.
        i = tf.range(d_model)[tf.newaxis, :]
        position = tf.range(position)[:, tf.newaxis]
        angles = position*tf.math.exp(-2*(i//2)*tf.math.log(10000.0)/d_model)

        # Compute sines on even and cosines on odd angle indices.
        sines = tf.math.sin(angles[:, 0::2])
        cosines = tf.math.cos(angles[:, 1::2])

        encoding = tf.concat([sines, cosines], -1)
        self._encoding = encoding[tf.newaxis, ...]

    def call(self, inputs):
        """Adds positional encoding to the inputs.

        Args:
            inputs: a Tensor with shape [batch_size, length, d_model].

        Returns:
            A Tensor with shape [batch_size, length, d_model], the
            inputs with positional encoding added.

        """
        # Add the positional encoding to the input tensor.
        return inputs + self._encoding[:, :tf.shape(inputs)[1], :]
