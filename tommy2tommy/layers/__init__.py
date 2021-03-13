import tensorflow as tf

from tensorflow.keras.utils import get_custom_objects
from tommy2tommy.layers.activations import gelu
from tommy2tommy.layers.activations import mish
from tommy2tommy.layers.activations import swish

# Register the custom activation functions to be able to reference them by string.
get_custom_objects().update({
    'gelu': tf.keras.layers.Activation(gelu),
    'mish': tf.keras.layers.Activation(mish),
    'swish': tf.keras.layers.Activation(swish),
})
