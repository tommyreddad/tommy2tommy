"""Rectified Adam optimizer.

Todo:
    * Implement _resource_apply_sparse and test it.

"""

import tensorflow as tf


class RAdam(tf.keras.optimizers.Optimizer):
    """Implements the Rectified Adam optimizer.

    The instability of the Adam optimizer during early stages is
    identified through the large variance of the adaptive learning
    rate. In Rectified Adam, this is controlled by finding the length
    of a simple moving average corresponding to the exponential moving
    average for the gradient second moment estimate. If this length is
    short, this means that the adaptive learning rate relies on few
    samples, and therefore is subject to higher risk of
    instability. In this case, Rectified Adam ignores the adaptive
    rate, but otherwise employs the ordinary Adam algorithm.

    Args:
        learning_rate (float, optional): the learning rate of the
            optimizer.
        beta_1 (float, optional): momentum coefficient for the
            gradient first moment estimate.
        beta_2 (float, optional): momentum coefficient for the
            gradient second moment estimate.
        epsilon (float, optional): small quantity for numerical
            stability.
        sma_threshold (float, optional): the lower threshold for the
            length of the simple moving average before using a fixed
            learning rate.
        name (string, optional): the name of the optimizer.
        **kwargs (dict): named arguments of the parent class.

    Reference:
        [Liu et al., 2020](https://arxiv.org/pdf/1908.03265.pdf)

    """

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 sma_threshold=4.5,
                 name="RAdam",
                 **kwargs):
        super(RAdam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('sma_threshold', sma_threshold)
        self._epsilon = epsilon or tf.keras.backend.epsilon()

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype

        # Load hyperparameters and important quantities.
        lr_t = self._decayed_lr(var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        sma_threshold = self._get_hyper('sma_threshold', var_dtype)
        epsilon_t = tf.convert_to_tensor(self._epsilon, var_dtype)

        # Load slots.
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # m_t <- beta1*m + (1 - beta1)*g_t
        m_t = beta_1_t*m + (1.0 - beta_1_t)*grad
        m_t = m.assign(m_t, use_locking=self._use_locking)
        m_hat_t = m_t/(1.0 - beta_1_power)

        # v_t <- beta2*v + (1 - beta2)*g_t*g_t
        v_t = beta_2_t*v + (1.0 - beta_2_t)*tf.square(grad)
        v_t = v.assign(v_t, use_locking=self._use_locking)
        denom_t = epsilon_t + tf.sqrt(v_t/(1.0 - beta_2_power))

        # Compute the length of the SMA estimate and rectification factor.
        sma_infty = 2.0/(1.0 - beta_2_t) - 1.0
        sma_t = sma_infty - 2.0*local_step*beta_2_power/(1.0 - beta_2_power)
        rect_t = tf.sqrt((sma_t - 4.0)*(sma_t - 2.0)*sma_infty /
                         ((sma_infty - 4.0)*(sma_infty - 2.0)*sma_t))

        # Only perform rectification if the SMA length estimate is small.
        var_t = tf.where(sma_t > sma_threshold, rect_t *
                         m_hat_t/denom_t, m_hat_t)
        var_update = var.assign_sub(lr_t*var_t, use_locking=self._use_locking)
        return tf.group(var_update, m_t, v_t)

    def get_config(self):
        config = super(RAdam, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'sma_threshold': self._serialize_hyperparameter('sma_threshold'),
            'epsilon': self._epsilon,
        })
        return config
