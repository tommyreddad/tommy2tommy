# -*- coding: utf-8 -*-
"""Transformer models.

This module contains different neural network models for transformers,
including the transformer encoder (e.g., BERT), transformer decoder
(e.g., GPT-2), and full transformer.

"""

import tensorflow as tf

from tommy2tommy.layers import bias
from tommy2tommy.layers.transformer import encoder
from tommy2tommy.layers.transformer import decoder


class TransformerEncoder(tf.keras.Model):
    """A transformer encoder model.

    Args:
        config (dict): model configuration hyperparameters.
        padding_id (int, optional): the padding token.
        **kwargs (dict): named arguments for the parent class.

    """

    def __init__(self, config, padding_id=0, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        # Set up the bias layer.
        self._padding_bias = bias.PaddingBias(padding_id, name="padding_bias")

        # Set up the inner encoder.
        self._encoder = encoder.TransformerEncoder(
            vocab_size=config['vocab_size'],
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            d_filter=config['d_filter'],
            num_heads=config['num_heads'],
            dropout_rate=config['dropout_rate'],
            ffn_activation=config['ffn_activation'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            name="encoder")

    def call(self, inputs, training=False):
        """Apply the encoder to batched inputs.

        Args:
            inputs: a Tensor with shape [batch_size, length].
            training (boolean, optional): training if true, else inferring.

        Returns:
            A Tensor with shape [batch_size, length, d_model], the
            encoded input.

        """
        padding_bias = self._padding_bias(inputs)
        outputs = self._encoder(inputs, padding_bias, training=training)
        return outputs


class TransformerDecoder(tf.keras.Model):
    """A transformer decoder model.

    Args:
        config (dict): model configuration hyperparameters.
        padding_id (int, optional): the padding token.
        **kwargs (dict): named arguments of the parent class.

    """

    def __init__(self, config, padding_id=0, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)

        # Set up the bias layers.
        self._padding_bias = bias.PaddingBias(padding_id, name="padding_bias")
        self._causal_bias = bias.CausalBias(name="causal_bias")

        # Set up the inner decoder.
        self._decoder = decoder.TransformerDecoder(
            vocab_size=config['vocab_size'],
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            d_filter=config['d_filter'],
            num_heads=config['num_heads'],
            dropout_rate=config['dropout_rate'],
            ffn_activation=config['ffn_activation'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            encoder_decoder=False,
            name="decoder")

    def call(self, inputs, training=False):
        """Apply the decoder to batched inputs.

        Args:
            inputs: a Tensor with shape [batch_size, length].
            training (boolean, optional): true if training, else inferring.

        Returns:
            A Tensor with shape [batch_size, length, d_model], the
            decoded input.

        """
        padding_bias = self._padding_bias(inputs)
        causal_bias = self._causal_bias(inputs)
        decoder_bias = tf.minimum(padding_bias, causal_bias)
        outputs = self._decoder(inputs, decoder_bias, training=training)
        return outputs


class TransformerLM(tf.keras.Model):
    """A transformer language model. Just contains the decoder part of the
    transformer.

    Args:
        config (dict): model configuration hyperparameters.
        padding_id (int, optional): the padding token.
        **kwargs (dict): named arguments of the parent class.

    """

    def __init__(self, config, padding_id=0, **kwargs):
        super(TransformerLM, self).__init__(**kwargs)

        # Set up the bias layers.
        self._padding_bias = bias.PaddingBias(padding_id, name="padding_bias")
        self._causal_bias = bias.CausalBias(name="causal_bias")

        # Set up the inner decoder.
        self._decoder = decoder.TransformerDecoder(
            vocab_size=config['vocab_size'],
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            d_filter=config['d_filter'],
            num_heads=config['num_heads'],
            dropout_rate=config['dropout_rate'],
            ffn_activation=config['ffn_activation'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            encoder_decoder=False,
            name="decoder")
        self._to_out = tf.keras.layers.Dense(config['vocab_size'], name="out")

    def call(self, inputs, training=False):
        """Apply the decoder to batched inputs.

        Args:
            inputs: a Tensor with shape [batch_size, length].
            training (boolean, optional): true if training, else inferring.

        Returns:
            A Tensor with shape [batch_size, length, d_model], the
            decoded input.

        """
        padding_bias = self._padding_bias(inputs)
        causal_bias = self._causal_bias(inputs)
        decoder_bias = tf.minimum(padding_bias, causal_bias)
        outputs = self._decoder(inputs, decoder_bias, training=training)
        outputs = self._to_out(outputs)
        return outputs


class Transformer(tf.keras.Model):
    """A full transformer model, including encoder and decoder.

    Args:
        config (dict): model configuration hyperparameters.
        padding_id (int, optional): the padding token.
        **kwargs (dict, optional): named arguments of the parent class.

    """

    def __init__(self, config, padding_id=0, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        # Set up the bias layers.
        self._padding_bias = bias.PaddingBias(padding_id, name="padding_bias")
        self._causal_bias = bias.CausalBias(name="causal_bias")

        # Set up the inner encoder and decoder and output layer.
        self._encoder = encoder.TransformerEncoder(
            vocab_size=config['vocab_size'],
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            d_filter=config['d_filter'],
            num_heads=config['num_heads'],
            dropout_rate=config['dropout_rate'],
            ffn_activation=config['ffn_activation'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            name="encoder")
        self._decoder = decoder.TransformerDecoder(
            vocab_size=config['vocab_size'],
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            d_filter=config['d_filter'],
            num_heads=config['num_heads'],
            dropout_rate=config['dropout_rate'],
            ffn_activation=config['ffn_activation'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            encoder_decoder=True,
            name="decoder")
        self._to_out = tf.keras.layers.Dense(config['output_vocab_size'])

    def call(self, encoder_inputs, decoder_inputs, training=False):
        """Apply the transformer to batched inputs.

        Args:
            encoder_inputs: a Tensor with shape [batch_size, length].
            decoder_inputs: a Tensor with shape [batch_size, length].
            training (boolean, optional): training if true, else inferring.

        Returns:
            A Tensor with shape [batch_size, length, output_vocab_size].

        """
        # Compute the encoder outputs.
        encoder_padding_bias = self._padding_bias(encoder_inputs)
        encoder_outputs = self._encoder(encoder_inputs, encoder_padding_bias,
                                        training=training)

        # Compute the decoder outputs, feed in the encoder outputs.
        decoder_padding_bias = self._padding_bias(decoder_inputs)
        decoder_causal_bias = self._causal_bias(decoder_inputs)
        decoder_bias = tf.minimum(decoder_padding_bias, decoder_causal_bias)
        decoder_outputs = self._decoder(decoder_inputs=decoder_inputs,
                                        decoder_bias=decoder_bias,
                                        encoder_outputs=encoder_outputs,
                                        encoder_bias=encoder_padding_bias,
                                        training=training)
        output = self._to_out(decoder_outputs)
        return output
