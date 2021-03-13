"""Decoding search algorithms.

This module contains algorithms for searching for probable decoded
sequences. Common solutions include greedy search and beam search.

Todo:
    * Implement beam search.
    * Implement beam stack search.

"""

import tensorflow as tf


def greedy_search(model_fn, prefix, length):
    """Greedy search for an output sequence.

    Args:
        model_fn: a function which takes as input a Tensor with shape
            [batch_size, length], the decoded output so far, and
            produces a Tensor of logits with shape [batch_size,
            length, output_vocab_size] of future outputs conditioned
            on the currently decoded output.
        prefix: a Tensor with shape [batch_size, prefix_length], the
            prefix fed into the model before beginning the search.
        length (int): the maximum length of output sequences.

    Returns:
        A Tensor with shape [batch_size, length], the greedily decoded
        output sequences.

    """
    shape = tf.shape(prefix)
    batch_size = shape[0]
    prefix_length = shape[1]
    curr_inputs = tf.zeros(shape=(batch_size, 1), dtype=tf.int32)
    for i in range(length):
        predictions = model_fn(curr_inputs, training=False)

        # Select the next predicted token logits.
        predictions = predictions[:, -1]

        # Greedy search picks the most likely token for each index.
        prediction_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if i < prefix_length:
            prediction_id = prefix[:, i]
        prediction_id = prediction_id[..., tf.newaxis]

        curr_inputs = tf.concat([curr_inputs, prediction_id], -1)
    # Disregard the initial padding token.
    return curr_inputs[:, 1:]
