import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import scipy.signal
from command import Commander
from config import Config


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []

    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class ACNetwork:
    def __init__(self, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None]+Config.STATE_SHAPE, dtype=tf.float32)  # grayscale or RGB?
            self.aux_depth_labels = [tf.placeholder(shape=[None] + [8], dtype=tf.float32) for i in range(4*16)]
            self.conv1 = slim.conv2d(inputs=self.inputs,
                                     num_outputs=16,
                                     kernel_size=[8, 8],
                                     stride=[4, 4],
                                     padding='VALID',
                                     activation_fn=tf.nn.elu)
            self.conv2 = slim.conv2d(inputs=self.conv1,
                                     num_outputs=32,
                                     kernel_size=[4, 4],
                                     stride=[2, 2],
                                     padding='VALID',
                                     activation_fn=tf.nn.elu)
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

            # LSTM
            lstm_cell = rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            # self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.inputs)[:1]
            self.state_in = rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = \
                tf.nn.dynamic_rnn(lstm_cell,
                                  rnn_in,
                                  sequence_length=step_size,
                                  initial_state=self.state_in,
                                  time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            self.policy = slim.fully_connected(rnn_out, Config.ACTIONS,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None
                                              )
            if Config.AUX_TASK_D2:
                self.aux_depth2_hidden = slim.fully_connected(rnn_out, 128, activation_fn=tf.nn.elu)
                self.aux_depth2_logits = [
                    slim.fully_connected(self.aux_depth2_hidden, 8, activation_fn=None)  # , scope='d2_logits'
                    for i in range(4*16)]

            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, Config.ACTIONS, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                loss_array = [0.5 * self.value_loss, -0.01 * self.entropy, self.policy_loss]

                # Auxiliary Loss Functions
                if Config.AUX_TASK_D2:
                    self.aux_depth2_losses = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.aux_depth_labels[i], logits=self.aux_depth2_logits[i])) for i in range(4 * 16)]
                    self.aux_depth2_loss = tf.add_n(self.aux_depth2_losses)
                    loss_array.append(1.0 * self.aux_depth2_loss)

                self.loss = tf.add_n(loss_array)

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                if trainer is not None:
                    self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker:
    def __init__(self, name, trainer, global_episodes):
        self.name = 'worker_' + str(name)
        self.number = name
        self.model_path = Config.MODEL_PATH
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter('train' + str(self.number), graph=tf.get_default_graph())

        # restructuring
        self.env = Commander(self.number, mode='test')

        self.local_AC = ACNetwork(self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        # setting up game here
        # self.env = Commander(self.client, sim_dir, self.sim)  # UnrealCV controller
        self.actions = self.env.action_space

    def train(self, rollout, bootstrap_value, gamma, sess):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]
        aux_depth = rollout[:, 6]
        identity = np.eye(8)

        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(rewards_plus, gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]   # GAE?
        advantages = discount(advantages, gamma)

        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(np.expand_dims(obs, 0) for obs in observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]
                     }

        ops_for_run = [self.local_AC.value_loss,
                       self.local_AC.policy_loss,
                       self.local_AC.entropy,
                       self.local_AC.grad_norms,
                       self.local_AC.var_norms,
                       self.local_AC.apply_grads]

        if Config.AUX_TASK_D2:
            depth_list = [np.vstack(identity[batch, :] for batch in px) for px in np.transpose(aux_depth)]
            depth_labels = np.swapaxes(np.array(depth_list), 0, 1)
            feed_dict.update({self.local_AC.aux_depth_labels[px]: depth_labels[px] for px in range(4*16)})
            ops_for_run.insert(3, self.local_AC.aux_depth2_loss)
            v_l, p_l, e_l, ad2_l, g_n, v_n, _ = sess.run(ops_for_run, feed_dict=feed_dict)
        else:
            v_l, p_l, e_l, g_n, v_n, _ = sess.run(ops_for_run, feed_dict=feed_dict)

        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print('Starting worker ' + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                self.env.new_episode()
                s = self.env.get_observation()
                episode_frames.append(s)
                aux_depth = np.expand_dims(self.env.get_observation(viewmode='depth').flatten(), 0)
                # s = process_frame(s)
                rnn_state = self.local_AC.state_init

                while self.env.is_episode_finished() is False:
                    a_dist, v, rnn_state = sess.run([self.local_AC.policy,
                                                     self.local_AC.value,
                                                     self.local_AC.state_out],
                                                    feed_dict={self.local_AC.inputs: [s],
                                                               self.local_AC.state_in[0]: rnn_state[0],
                                                               self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    r = self.env.action(self.actions[a])
                    d = self.env.is_episode_finished()
                    if d is False:
                        s1 = self.env.get_observation()
                        episode_frames.append(s1)
                        # s1 = process_frame(s1)
                    else:
                        s1 = s
                    episode_buffer.append([s, a, r, s1, d, v[0, 0], aux_depth])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    aux_depth = np.expand_dims(self.env.get_observation(viewmode='depth').flatten(), 0)

                    if (len(episode_buffer) == 30) and (d is not True) and (episode_step_count != Config.MAX_EPISODE_LENGTH):
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, v1, Config.GAMMA, sess)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d or (episode_step_count == Config.MAX_EPISODE_LENGTH - 1):
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, 0.0, Config.GAMMA, sess)

                if episode_count % 5 == 0 and episode_step_count != 0:
                    if episode_count % Config.MODEL_SAVE_FREQ == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.cptk')

                    if Config.VERBOSITY == 1:
                        if episode_count % 100 == 0:
                            print('[{}] completed {} episodes.'.format(self.name, episode_count))

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                    if episode_count > Config.MAX_EPISODES:
                        coord.request_stop()
                episode_count += 1

            # shutting down client and sim
            print('Shutting down {}...    '.format(self.name))
            self.env.shut_down()
            print('       Sim and client closed.')
