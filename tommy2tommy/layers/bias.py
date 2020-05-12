# -*- coding: utf-8 -*-
"""Bias layers for attention logits.

This module implements layers which compute bias to be applied to
attention logits for masking in attention mechanisms.

Todo:
    * Implement Reformer causal attention bias.
    * Implement local attention bias.
    * Implement proximal attention bias.

"""

import tensorflow as tf


class CausalBias(tf.keras.layers.Layer):
    """Compute causal bias for batched input sequences."""

    def call(self, inputs):
        """Compute the bias for specific inputs.

        Args:
            inputs: a Tensor with shape [batch_size, length].

        Returns:
            A Tensor with shape [1, 1, length, length], implementing
            causal bias.

        """
        length = tf.shape(inputs)[-1]
        mask = tf.linalg.band_part(tf.ones(shape=(length, length)), -1, 0)
        return -1.0e9*(1.0 - mask[tf.newaxis, tf.newaxis, :, :])


class PaddingBias(tf.keras.layers.Layer):
    """Compute padding bias for batched input sequences.

    Args:
        padding_id (int, optional): value of the padding tokens in the
            input.

    """

    def __init__(self, padding_id=0, **kwargs):
        super(PaddingBias, self).__init__(**kwargs)
        self._padding_id = padding_id

    def call(self, inputs):
        """Compute padding bias for specific inputs.

        Args:
            inputs: a Tensor with shape [batch_size, length].

        Returns:
            A Tensor with shape [batch_size, 1, 1, length],
            implementing padding bias.

        """
        inverse_mask = tf.cast(tf.equal(inputs, self._padding_id), tf.float32)
        return -1.0e9*inverse_mask[:, tf.newaxis, tf.newaxis, :]
