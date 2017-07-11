import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network():
    def __init__(self):
        
        action_space_size = 3

        self.inputs = tf.placeholder(shape=[None, 84, 84, 3], dtype=tf.float32)  # grayscale or RGB?
        self.conv1 = slim.conv2d(inputs=self.inputs,
                                 num_outputs=16,
                                 kernel_size=[8, 8],
                                 stride=[4, 4],
                                 padding='VALID',
                                 activation_fn=tf.nn.elu)
        self.conv2 = slim.conv2d(inputs=self.conv2,
                                 num_outputs=32,
                                 kernel_size=[4, 4],
                                 stride=[2, 2],
                                 padding='VALID',
                                 activation_fn=tf.nn.elu)
        hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

        # LSTM
        lstm_cell = rnn.BasicLSTMCell(256, state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), tf.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), tf.float32)
        # self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        # self.state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(hidden, [0])
        step_size = tf.shape(self.inputs)[:1]
        state_in = rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                                      rnn_in,
                                                      sequence_length=step_size,
                                                      initial_state=state_in,
                                                      time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        self.policy = slim.fully_connected(rnn_out, action_space_size,
                                           activation_fn=tf.nn.softmax,
                                           weights_initializer=normalized_columns_initializer(0.01),
                                           biases_initializer=None)
        self.value = slim.fully_connected(rnn_out, 1,
                                          activation_fn=None,
                                          weights_initializer=normalized_columns_initializer(),
                                          biases_initializer=None
                                          )







