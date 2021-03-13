"""Activation functions.

This module implements some alternative activation functions which are
not present in vanilla TensorFlow.

"""

import math

import tensorflow as tf


@tf.function
def gelu(x):
    """Applies the approximated GELU to the input.

    The Gaussian Error Linear Unit (GELU), as described by [Hendrycks
    and Gimpel, 2018](https://arxiv.org/abs/1606.08415), is a smoothed
    version of the ReLU.

    Args:
        x: a float value or Tensor. The input of the activation.

    Returns:
        A float Tensor with GELU applied.

    """
    return 0.5*x*(1.0 + tf.math.tanh(math.sqrt(2.0/math.pi)*(x + 0.044715*x*x*x)))


@tf.function
def mish(x):
    """Applies the Mish activation to the input.

    See [Misra, 2019](https://arxiv.org/abs/1908.08681). The Mish
    activation is another smoothed ReLU. Some have reported
    improvements from using Mish over other ReLU alternatives.

    Args:
        x: a float value or Tensor. The input of the activation.

    Returns:
        A float Tensor with Mish applied.

    """
    return x*tf.math.tanh(tf.math.softplus(x))


@tf.function
def swish(x, beta=1.0):
    """Applies the Swish activation to the input.

    See [Ramachandran et al.,
    2017](https://arxiv.org/abs/1710.05941). Swish is another smoothed
    alternative to ReLU, discovered through a reinforcement learning
    search of possible activation functions.

    Args:
        x: a float value or Tensor. The input of the activation.
        beta (float, optional): a higher value corresponds to a
            tighter approximation to ReLU.

    Returns:
        A float Tensor with Swish applied.

    """
    return x*tf.math.sigmoid(x*beta)
