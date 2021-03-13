
"""Yogi optimizer.

Todo:
    * Implement _resource_apply_sparse and test it.

"""

import tensorflow as tf


class Yogi(tf.keras.optimizers.Optimizer):
    """Implements the Yogi optimizer.

    Adam struggles from adaptive learning rate instability in
    particular from the "multiplicative" nature of the gradient second
    moment exponential moving average estimate. Yogi replaces this
    with an update which is more additive in nature. For good results,
    Yogi should be used with larger values of epsilon and rougly 10x
    the learning rate of Adam.

    Args:
        learning_rate (float, optional): the learning rate of the
            optimizer.
        beta_1 (float, optional): momentum coefficient for the
            gradient first moment estimate.
        beta_2 (float, optional): momentum coefficient for the
            gradient second moment estimate.
        epsilon (float, optional): small quantity for numerical
            stability.
        activation (string, optional): the sign-like activation in the
            gradient second moment estimate update.
        name (string, optional): the name of the optimizer.
        **kwargs (dict): named arguments of the parent class.

    Reference:
        [Zaheer et al., 2018](https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf)

    """

    def __init__(self,
                 learning_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-5,
                 activation="sign",
                 name="Yogi",
                 **kwargs):
        super(Yogi, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._epsilon = epsilon or tf.keras.backend.epsilon()
        self._activation = activation

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
        epsilon_t = tf.convert_to_tensor(self._epsilon, var_dtype)
        activation = self._activation

        # Load slots.
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # m_t <- beta1*m + (1 - beta1)*g_t
        m_t = beta_1_t*m + (1.0 - beta_1_t)*grad
        m_t = m.assign(m_t, use_locking=self._use_locking)
        m_hat_t = m_t/(1.0 - beta_1_power)

        # v_t <- v + (1 - beta2)*sign(g_t*g_t - v)*g_t*g_t
        grad_squared = tf.square(grad)
        if activation == "sign":
            sign = tf.math.sign(grad_squared - v)
        elif activation == "tanh":
            sign = tf.math.tanh(10.0*(grad_squared - v))
        elif activation == "hardtanh":
            sign = tf.math.maximum(-1.0, tf.math.minimum(1.0,
                                                         10.0*(grad_squared - v)))
        else:
            raise NotImplementedError(
                "Unknown activation in Yogi optimizer: {}".format(activation))
        v_t = v + (1 - beta_2_t)*sign*grad_squared
        v_t = v.assign(v_t, use_locking=self._use_locking)
        denom_t = epsilon_t + tf.sqrt(v_t/(1.0 - beta_2_power))

        # Variable update.
        var_update = var.assign_sub(
            lr_t*m_hat_t/denom_t, use_locking=self._use_locking)
        return tf.group(var_update, m_t, v_t)

    def get_config(self):
        config = super(Yogi, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self._epsilon,
            'activation': self._activation,
        })
        return config
