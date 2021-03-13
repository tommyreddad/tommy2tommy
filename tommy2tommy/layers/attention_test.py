import tensorflow as tf
import numpy as np
from tommy2tommy.layers import attention


class TestSplitHeads(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 2
        num_heads = 2
        length = 10
        d_model = 4
        x = tf.ones([batch_size, length, d_model])

        split_heads = attention.SplitHeads(num_heads=num_heads)
        output = split_heads(x)
        self.assertShapeEqual(
            np.zeros((batch_size, num_heads, length, d_model//num_heads)), output)

    def test_output_value(self):
        batch_size = 1
        num_heads = 2
        length = 3
        d_model = 2
        x = tf.ones([batch_size, length, d_model])
        expected_output = tf.ones(
            [batch_size, num_heads, length, d_model//num_heads])

        split_heads = attention.SplitHeads(num_heads=num_heads)
        output = split_heads(x)
        self.assertAllEqual(expected_output, output)


class TestMergeHeads(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 2
        num_heads = 2
        length = 10
        d_model = 4
        x = tf.ones([batch_size, num_heads, length, d_model//num_heads])

        merge_heads = attention.MergeHeads()
        output = merge_heads(x)
        self.assertShapeEqual(
            np.zeros((batch_size, length, d_model)), output)

    def test_output_value(self):
        batch_size = 1
        num_heads = 2
        length = 3
        d_model = 2
        x = tf.ones([batch_size, num_heads, length, d_model//num_heads])
        expected_output = tf.ones(
            [batch_size, length, d_model])

        merge_heads = attention.MergeHeads()
        output = merge_heads(x)
        self.assertAllEqual(expected_output, output)


class TestDotProductionAttention(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 2
        length_q = 3
        depth_qk = 4
        length_kv = 5
        depth_v = 6

        q = tf.ones([batch_size, length_q, depth_qk])
        k = tf.ones([batch_size, length_kv, depth_qk])
        v = tf.ones([batch_size, length_kv, depth_v])

        dot_product_attention = attention.DotProductAttention()
        output, _ = dot_product_attention(q, k, v)
        self.assertShapeEqual(
            np.zeros((batch_size, length_q, depth_v)), output)

    def test_output_value(self):
        batch_size = 1
        length_q = 2
        depth_qk = 2
        length_kv = 2
        depth_v = 2

        q = tf.ones([batch_size, length_q, depth_qk])
        k = tf.ones([batch_size, length_kv, depth_qk])
        v = tf.ones([batch_size, length_kv, depth_v])
        expected_output = np.array([[[1, 1], [1, 1]]])

        dot_product_attention = attention.DotProductAttention()
        output, _ = dot_product_attention(q, k, v)
        self.assertAllEqual(expected_output, output)

    def testOutputValueWithBias(self):
        pass


class TestShiftRight(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 2
        length = 10
        d_model = 4
        x = tf.ones([batch_size, length, d_model])

        shift_right = attention.ShiftRight()
        output = shift_right(x)
        self.assertShapeEqual(np.zeros((batch_size, length, d_model)), output)

    def test_output_value_training(self):
        batch_size = 1
        length = 5
        d_model = 1
        x = tf.ones([batch_size, length, d_model])
        expected_output_training = np.array([[[0], [1], [1], [1], [1]]])
        expected_output_no_training = np.array([[[1], [1], [1], [1], [1]]])

        shift_right = attention.ShiftRight()
        output_training = shift_right(x, training=True)
        self.assertAllEqual(expected_output_training, output_training)
        output_no_training = shift_right(x, training=False)
        self.assertAllEqual(expected_output_no_training, output_no_training)


class TestMultiheadAttention(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 2
        num_heads = 2
        length = 5
        d_model = 4

        q = tf.ones([batch_size, length, d_model])
        k = tf.ones([batch_size, length, d_model])
        v = tf.ones([batch_size, length, d_model])

        multihead_attention = attention.MultiheadAttention(
            d_model=d_model, num_heads=num_heads)
        output, _ = multihead_attention(q, k, v)
        self.assertShapeEqual(np.zeros((batch_size, length, d_model)), output)


if __name__ == "__main__":
    tf.test.main()
