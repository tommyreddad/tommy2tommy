"""Transformer encoder layers.

This module contains layers relevant to transformer encoders. For full
models, use tommy2tommy.models.transformer instead.

"""

import tensorflow as tf

from tommy2tommy.layers import attention
from tommy2tommy.layers import embeddings


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """A transformer encoder block, containing one self-attention
    mechanism and one feedforward block.

    Args:
        d_model (int): the hidden size of the block.
        d_filter (int): the width of the feedforward layer.
        num_heads (int): the number of attention heads in the
            multihead attention mechanism.
        dropout_rate (float): the dropout rate.
        ffn_activation: the activation function to be used in the
            feedforward layer, should be a string or point directly to the
            function, as in the activation parameter for
            tf.keras.layers.Dense.
        layer_norm_epsilon (float): epsilon parameter for the layer
            normalization.
        **kwargs (dict): named arguments for the parent class.

    """

    def __init__(self, d_model, d_filter, num_heads, dropout_rate,
                 ffn_activation, layer_norm_epsilon, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)

        self._self_attn = attention.MultiheadAttention(
            d_model, num_heads, name="self_attn")
        self._self_attn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._self_attn_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="self_attn_layer_norm")

        self._ffn_filter = tf.keras.layers.Dense(
            d_filter, name="ffn_layer", activation=ffn_activation)
        self._ffn_hidden = tf.keras.layers.Dense(d_model, name="ffn_layer")
        self._ffn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._ffn_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="ffn_layer_norm")

    def call(self, inputs, bias, training=False):
        """Computes the output of the encoder block.

        Args:
            inputs: a Tensor with shape [batch_size, length, d_model].
            bias: a Tensor with shape broadcastable to [batch_size,
                length, length].        
            training (boolean, optional): training is true, else inferring.

        Returns:
            A Tensor with shape [batch_size, length, d_model], the
            output of the encoder block for the given inputs.

        """
        # Compute the attention.
        x, _ = self._self_attn(query=inputs, key=inputs, value=inputs,
                               bias=bias, training=training)

        # Apply dropout and layer normalization after adding residual.
        x = self._self_attn_dropout(x, training=training)
        y = self._self_attn_layer_norm(inputs + x)

        # Compute the feedforward part.
        x = self._ffn_filter(y, training=training)
        x = self._ffn_hidden(x, training=training)

        # Apply dropout and layer normalization after adding residual.
        x = self._ffn_dropout(x, training=training)
        y = self._ffn_layer_norm(y + x)

        return y


class TransformerEncoder(tf.keras.layers.Layer):
    """A transformer encoder.

    Args:
        vocab_size (int): the size of the input vocabulary.
        num_layers (int): the number of attention layers in the encoder.
        d_model (int): the hidden size of the network.
        d_filter (int): the width of the feedforward layers.
        num_heads (int): the number of attention heads in the
            multihead attention mechanisms.
        dropout_rate (float): the dropout rate.
        ffn_activation: the activation function to be used in the
            feedforward layers, should be a string or point directly
            to the function, as in the activation parameter for
            tf.keras.layers.Dense.
        layer_norm_epsilon (float): epsilon parameter for the layer
            normalization.
        **kwargs (dict): named arguments for the parent class.

    """

    def __init__(self, vocab_size, num_layers, d_model, d_filter,
                 num_heads, dropout_rate, ffn_activation,
                 layer_norm_epsilon, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        # Set up the embedding layer.
        self._embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self._positional_encoding = embeddings.PositionalEncoding()
        self._dropout = tf.keras.layers.Dropout(dropout_rate)

        # Set up the encoder layers.
        self._layers = [TransformerEncoderBlock(d_model=d_model,
                                                d_filter=d_filter,
                                                num_heads=num_heads,
                                                dropout_rate=dropout_rate,
                                                ffn_activation=ffn_activation,
                                                layer_norm_epsilon=layer_norm_epsilon,
                                                name="block_{}".format(i))
                        for i in range(num_layers)]

    def call(self, inputs, bias, training=False):
        """Computes the output of the encoder.

        Args:
            inputs: a Tensor with shape [batch_size, length].
            bias: a Tensor with shape broadcastable to [batch_size,
                length, length].
            training (boolean, optional): training if true, else inferring.

        Returns:
            A Tensor with shape [batch_size, length, d_model], the
            output of the encoder for the given inputs.

        """
        # Embed the inputs before computing.
        x = self._embedding(inputs)
        x = self._positional_encoding(x)
        x = self._dropout(x, training=training)

        # Input is fed from one block to the next.
        for block in self._layers:
            x = block(x, bias, training=training)
        return x
