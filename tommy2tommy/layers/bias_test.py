import tensorflow as tf
import numpy as np
from tommy2tommy.layers import bias


class TestCausalBias(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 2
        length = 10
        x = tf.ones([batch_size, length])

        causal_bias = bias.CausalBias()
        output = causal_bias(x)
        self.assertShapeEqual(np.zeros((1, 1, length, length)), output)

    def test_output_value(self):
        batch_size = 1
        length = 3
        x = tf.ones([batch_size, length])
        expected_output = np.array(
            [[[[-0, -1e9, - 1e9], [0, 0, -1e9], [0, 0, 0]]]])

        causal_bias = bias.CausalBias()
        output = causal_bias(x)
        self.assertAllEqual(expected_output, output)


class TestPaddingBias(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 2
        length = 10
        padding_id = -1
        x = tf.ones([batch_size, length])

        padding_bias = bias.PaddingBias(padding_id=padding_id)
        output = padding_bias(x)
        self.assertShapeEqual(np.zeros((batch_size, 1, 1, length)), output)

    def test_output_value(self):
        padding_id = -1
        x = np.array([[0, 0, -1]])
        expected_output = np.array([[[[0, 0, -1e9]]]])

        padding_bias = bias.PaddingBias(padding_id=padding_id)
        output = padding_bias(x)
        self.assertAllEqual(expected_output, output)


if __name__ == "__main__":
    tf.test.main()
