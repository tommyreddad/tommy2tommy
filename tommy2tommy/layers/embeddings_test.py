import tensorflow as tf
import numpy as np
from tommy2tommy.layers import embeddings


class TestPositionalEncoding(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 2
        length = 10
        d_model = 4
        x = tf.ones([batch_size, length, d_model])

        positional_encoding = embeddings.PositionalEncoding(max_length=10)
        output = positional_encoding(x)
        self.assertShapeEqual(np.zeros((batch_size, length, d_model)), output)


if __name__ == "__main__":
    tf.test.main()
