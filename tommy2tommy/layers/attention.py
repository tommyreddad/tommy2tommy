"""Attention.

This module contains functions and layers relevant to query-key-value
multihead attention, as used in the transformer.

"""

import tensorflow as tf


class SplitHeads(tf.keras.layers.Layer):
    """Split the Tensor of attention channels into values per head.

    Args:
        num_heads (int): the number of attention heads.
        **kwargs (dict): named arguments of the parent class.

    """

    def __init__(self, num_heads, **kwargs):
        super(SplitHeads, self).__init__(**kwargs)
        self._num_heads = num_heads

    def call(self, inputs):
        """Split the inputs.

        Args:
            inputs: a Tensor with shape [batch_size, length, d_model].

        Returns:
            A Tensor with shape [batch_size, num_heads, length,
                d_model/num_heads].

        """
        shape = tf.shape(inputs)
        new_shape = [shape[0], shape[1],
                     self._num_heads, shape[2]//self._num_heads]
        inputs = tf.reshape(inputs, new_shape)
        return tf.transpose(inputs, (0, 2, 1, 3))


class MergeHeads(tf.keras.layers.Layer):
    """The inverse of SplitHeads.

    Args:
        **kwargs (dict): named arguments of the parent class.
    """

    def call(self, inputs):
        """Merge the input attention heads into a lower-dimensional Tensor.

        Args:
            inputs: a Tensor with shape [batch_size, num_heads,
                length, d_model/num_heads]

        Returns:
            A Tensor with shape [batch_size, length, d_model].

        """
        inputs = tf.transpose(inputs, (0, 2, 1, 3))
        shape = tf.shape(inputs)
        new_shape = [shape[0], shape[1], shape[2]*shape[3]]
        return tf.reshape(inputs, new_shape)


class DotProductAttention(tf.keras.layers.Layer):
    """Single head dot-product attention layer with queries, keys, and
    values.

    Args:
        scaled (boolean, optional): if true, scales the Q-K matrix
            product for stability.

    """

    def __init__(self, scaled=True, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self._scaled = scaled

    def call(self, query, key, value, bias=None):
        """Compute attention with optional masking through added logit bias.

        Args:
            query: a Tensor with shape [..., length_q, depth_qk].
            key: a Tensor with shape [..., length_kv, depth_qk].
            value: a Tensor with shape [..., length_kv, depth_v].
            bias (optional): a Tensor with shape broadcastable to
                [..., length_q, length_kv].
            scaled (boolean, optional): if true, scales the Q-K matrix
                product for stability.

        Returns:
            A Tensor with shape [..., length_q, depth_v], the computed
                attention values.
            Also returns a Tensor with shape [..., length_q,
                length_kv], the attention weights.

        """
        # Compute the attention logits and apply bias for masking.
        logits = tf.matmul(query, key, transpose_b=True)
        if self._scaled:
            dim = tf.cast(tf.shape(key)[-1], tf.float32)
            logits /= tf.math.sqrt(dim)
        if bias is not None:
            logits += bias

        # Compute the attention and return.
        attn_weights = tf.nn.softmax(logits)
        attn = tf.matmul(attn_weights, value)
        return attn, attn_weights


class ShiftRight(tf.keras.layers.Layer):
    """Shifts the input to the right during training.

    Args:
        num_shifts (int, optional): the padding length.
        padding_id (int, optional): the padding value.
        **kwargs (dict): named arguments for the parent class.

    """

    def __init__(self, num_shifts=1, padding_id=0, **kwargs):
        super(ShiftRight, self).__init__(**kwargs)
        self._num_shifts = num_shifts
        self._padding_id = padding_id

    def call(self, inputs, training=False):
        """Apply the rightward shift.

        Args:
            inputs: a Tensor with shape [batch_size, length, d_model].
            training (boolean, optional): true if training, else inferring.

        Returns:
            A Tensor with shape [batch_size, length, d_model].

        """
        if not training:
            return inputs
        return tf.pad(inputs,
                      [[0, 0], [self._num_shifts, 0], [0, 0]],
                      constant_values=self._padding_id)[:, :-1, :]


class MultiheadAttention(tf.keras.layers.Layer):
    """A multihead attention mechanism. Can implement transformer
    self-attention or encoder-decoder attention, for example.

    Args:
        d_model (int): the hidden size of the attention mechanism.
        num_heads (int): the number of attention heads.
        **kwargs (dict): named arguments for the parent class.

    """

    def __init__(self, d_model, num_heads, scaled=True, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)

        if d_model % num_heads != 0:
            raise ValueError(
                'Attention dimension should be divisible by the number of heads.')

        # Set up the linear projection layers.
        self._to_q = tf.keras.layers.Dense(
            d_model, use_bias=False, name="to_q")
        self._to_k = tf.keras.layers.Dense(
            d_model, use_bias=False, name="to_k")
        self._to_v = tf.keras.layers.Dense(
            d_model, use_bias=False, name="to_v")
        self._to_out = tf.keras.layers.Dense(d_model, name="to_out")

        # Set up the attention layers.
        self._dot_product_attention = DotProductAttention(
            scaled=scaled, name="dot_product_attention")
        self._split_heads = SplitHeads(num_heads, name="split_heads")
        self._merge_heads = MergeHeads(name="merge_heads")

    def call(self, query, key, value, bias=None):
        """Apply multihead attention to queries, keys, and values.

        Args:
            query: a Tensor with shape [batch_size, length, d_model].
            key: a Tensor with shape [batch_size, length, d_model].
            value: a Tensor with shape [batch_size, length, d_model].
            bias (optional): a Tensor with shape broadcastable to
                [batch_size, length, length].

        Returns:
            A Tensor with shape [batch_size, length, d_model],
                attention applied to the input queries, keys, and
                values.
            Also returns a Tensor with shape [batch_size, num_heads,
                length, length], the attention weights.

        """
        # Linear projection for the inputs to Q, K, V matrices.
        q = self._to_q(query)
        k = self._to_k(key)
        v = self._to_v(value)

        # Split heads for multihead attention.
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Compute the scaled dot-product attention.
        attn, attn_weights = self._dot_product_attention(q, k, v, bias=bias)

        # Recombine the attention heads.
        attn = self._merge_heads(attn)

        # Linear projection to the output.
        out = self._to_out(attn)

        return out, attn_weights
