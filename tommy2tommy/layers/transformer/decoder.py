# -*- coding: utf-8 -*-
"""Transformer decoder layers.

This module contains layers relevant to transformer decoders. For full
models, use tommy2tommy.models.transformer instead.

"""

import tensorflow as tf

from tommy2tommy.layers import attention
from tommy2tommy.layers import embeddings


class TransformerDecoderBlock(tf.keras.layers.Layer):
    """A transformer decoder block, containing one self-attention
    mechanism, potentially encoder-decoder attention, and one
    feedforward block.

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
        encoder_decoder (boolean, optional): if true, consider
            encoder-decoder attention, otherwise ignore it.
        **kwargs (dict): named arguments for the parent class.

    """

    def __init__(self, d_model, d_filter, num_heads, dropout_rate,
                 ffn_activation, layer_norm_epsilon,
                 encoder_decoder=False, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)

        self._encoder_decoder = encoder_decoder

        self._self_attn = attention.MultiheadAttention(
            d_model, num_heads, name="self_attn")
        self._self_attn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._self_attn_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="self_attn_layer_norm")
        if encoder_decoder:
            self._encdec_attn = attention.MultiheadAttention(
                d_model, num_heads, name="endec_attn")
            self._encdec_attn_dropout = tf.keras.layers.Dropout(dropout_rate)
            self._encdec_attn_layer_norm = tf.keras.layers.LayerNormalization(
                epsilon=layer_norm_epsilon, name="encdec_attn_layer_norm")

        self._ffn_filter = tf.keras.layers.Dense(
            d_filter, name="ffn_layer", activation=ffn_activation)
        self._ffn_hidden = tf.keras.layers.Dense(d_model, name="ffn_layer")
        self._ffn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._ffn_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, name="ffn_layer_norm")

    def call(self, query_antecedent, key_antecedent, decoder_bias,
             encoder_bias=None, training=False):
        """Apply the decoder block to query and key inputs.

        Args:
            query_antecedent: a Tensor with shape [batch_size, length, d_model].
            key_antecedent: a Tensor with shape [batch_size, length, d_model].
            decoder_bias: a Tensor with shape broadcastable to
                [batch_size, length, length].
            encoder_bias: a Tensor with shape broadcastable to
                [batch_size, length, length].
            training (boolean, optional): training if true, else inferring.

        Returns:
            A Tensor with shape [batch_size, length, d_model].

        """
        # Compute the decoder self-attention.
        x, _ = self._self_attn(query=query_antecedent,
                               key=query_antecedent,
                               value=query_antecedent,
                               bias=decoder_bias, training=training)
        x = self._self_attn_dropout(x, training=training)
        y = self._self_attn_layer_norm(query_antecedent + x)

        if self._encoder_decoder:
            # Compute the encoder-decoder attention.
            x, _ = self._encdec_attn(query=y, key=key_antecedent,
                                     value=key_antecedent,
                                     bias=encoder_bias,
                                     training=training)

            # Apply layer normalization and dropout after adding residual.
            x = self._encdec_attn_dropout(x, training=training)
            y = self._encdec_attn_layer_norm(y + x)

        # Compute the feedforward part.
        x = self._ffn_filter(y, training=training)
        x = self._ffn_hidden(x, training=training)

        # Apply dropout and layer normalization after adding residual.
        x = self._ffn_dropout(x, training=training)
        y = self._ffn_layer_norm(y + x)

        return y


class TransformerDecoder(tf.keras.layers.Layer):
    """A transformer decoder.

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
        encoder_decoder (boolean, optional): if true, consider
            encoder-decoder attention, otherwise ignore it.
        **kwargs (dict): named arguments for the parent class.

    """

    def __init__(self, vocab_size, num_layers, d_model, d_filter,
                 num_heads, dropout_rate, ffn_activation,
                 layer_norm_epsilon, encoder_decoder=False, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)

        # Set up the embedding layer.
        self._embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self._shift_right = attention.ShiftRight()
        self._positional_encoding = embeddings.PositionalEncoding()
        self._dropout = tf.keras.layers.Dropout(dropout_rate)

        # Set up the decoder layers.
        self._layers = [TransformerDecoderBlock(d_model=d_model,
                                                d_filter=d_filter,
                                                num_heads=num_heads,
                                                dropout_rate=dropout_rate,
                                                ffn_activation=ffn_activation,
                                                layer_norm_epsilon=layer_norm_epsilon,
                                                encoder_decoder=encoder_decoder,
                                                name="block_{}".format(i))
                        for i in range(num_layers)]

    def call(self, decoder_inputs, decoder_bias, encoder_outputs=None,
             encoder_bias=None, training=False):
        """Apply the decoder to inputs, with potential encoder side-input.

        Args:
            decoder_inputs: a Tensor with shape [batch_size, length].
            decoder_bias: a Tensor with shape broadcastable to
                [batch_size, length, length].
            encoder_outputs (optional): a Tensor with shape
                [batch_size, length, d_model].
            encoder_bias: a Tensor with shape broadcastable to
                [batch_size, length, length].
            training (boolean, optional): training if true, else inferring.

        Returns:
            A Tensor with shape [batch_size, length, d_model], the
            outputs of the decoder when applied to the inputs.

        """

        # Embed the inputs before computing.
        x = self._embedding(decoder_inputs)
        x = self._shift_right(x, training=training)
        x = self._positional_encoding(x)
        x = self._dropout(x, training=training)

        # Input is fed from one block to the next.
        for block in self._layers:
            x = block(query_antecedent=x, decoder_bias=decoder_bias,
                      key_antecedent=encoder_outputs,
                      encoder_bias=encoder_bias, training=training)
        return x
