import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
import scipy.signal
import ucv_utils
from command import Commander


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

    """ Actor-Critic Network Class """

    def __init__(self, scope, trainer, config):

        # Graph Definition
        with tf.variable_scope(scope):
            # input image
            self.inputs = tf.placeholder(shape=[None]+config.STATE_SHAPE, dtype=tf.float32, name='input_image')

            # one-hot depth labels for each depth pixel
            self.aux_depth_labels = [tf.placeholder(shape=[None] + [8], dtype=tf.float32, name='depth_px_{}'.format(i)) for i in range(4*16)]

            # sin(heading_error)
            self.direction_input = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='heading_error_input')

            # angular velocity state
            self.velocity_state = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='velocity_state')

            # previous action
            self.prev_action = tf.placeholder(shape=[None, config.ACTIONS], dtype=tf.float32, name='previous_action')

            # previous reward
            self.prev_reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='previous_reward')

            # convolutional encoder
            self.conv1 = tf.contrib.layers.convolution2d(x=self.inputs,
                                                         num_output_channels=16,
                                                         kernel_size=[8, 8],
                                                         stride=[4, 4],
                                                         padding='VALID',
                                                         activation_fn=tf.nn.elu)
            self.conv2 = tf.contrib.layers.convolution2d(x=self.conv1,
                                                         num_output_channels=32,
                                                         kernel_size=[4, 4],
                                                         stride=[2, 2],
                                                         padding='VALID',
                                                         activation_fn=tf.nn.elu)

            shape = self.conv1.get_shape().as_list()
            flattened_dim = np.prod(shape[1:])
            flatten = tf.reshape(self.conv1, [-1, flattened_dim])
            hidden = tf.contrib.layers.fully_connected(x=flatten, num_output_units=256, activation_fn=tf.nn.elu)

            # Concatenating additional inputs with CNN outputs
            layers_to_concat = [hidden]
            if config.GOAL_ON:
                layers_to_concat.append(self.direction_input)
            if config.ACCELERATION_ACTIONS:
                layers_to_concat.append(self.velocity_state)
            if config.PREV_REWARD_ON:
                layers_to_concat.append(self.prev_reward)
            if config.PREV_ACTION_ON:
                layers_to_concat.append(self.prev_action)

            concatenated = tf.concat(concat_dim=1, values=layers_to_concat)
            rnn_in = tf.expand_dims(concatenated, [0])

            # LSTM layer
            lstm_cell = BasicLSTMCell(256)
            # c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            # h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = np.zeros((1, lstm_cell.state_size), np.float32)
            # c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            # h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size])
            step_size = tf.shape(self.inputs)[:1]
            lstm_outputs, lstm_state = \
                tf.nn.dynamic_rnn(lstm_cell,
                                  rnn_in,
                                  sequence_length=step_size,
                                  initial_state=self.state_in,
                                  time_major=False)
            # lstm_c, lstm_h = lstm_state
            # self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            self.state_out = lstm_state[:1, :]
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # output layers
            self.policy = tf.contrib.layers.fully_connected(x=rnn_out, num_output_units=config.ACTIONS,
                                                            activation_fn=tf.nn.softmax,
                                                            weight_init=normalized_columns_initializer(0.01),
                                                            bias_init=None)
            self.value = tf.contrib.layers.fully_connected(x=rnn_out, num_output_units=1,
                                                           activation_fn=None,
                                                           weight_init=normalized_columns_initializer(0.01),
                                                           bias_init=None)

            # auxiliary outputs
            if config.AUX_TASK_D2:
                self.aux_depth2_hidden = tf.contrib.layers.fully_connected(x=rnn_out, num_output_units=128,
                                                                           activation_fn=tf.nn.elu)
                self.aux_depth2_logits = [
                    tf.contrib.layers.fully_connected(
                        x=self.aux_depth2_hidden, num_output_units=8, activation_fn=None) for i in range(4*16)]

            # loss functions
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int64)
                self.actions_onehot = tf.one_hot(self.actions, config.ACTIONS, 1, 0)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * tf.cast(self.actions_onehot, dtype=tf.float32),
                                                         reduction_indices=[1])

                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                log_policy = tf.log(tf.clip_by_value(self.policy, 1e-20, 1))
                self.entropy = -tf.reduce_sum(self.policy * log_policy)
                self.policy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs, 1e-20, 1))
                                                  * self.advantages)
                loss_array = [0.5 * self.value_loss, -0.01 * self.entropy, self.policy_loss]

                # Auxiliary Loss Functions
                if config.AUX_TASK_D2:
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

    """ A3C agent, optionally augmented with aux tasks """

    def __init__(self, name, trainer, global_episodes, global_steps, logger_steps, config):
        self.name = 'worker_' + str(name)
        self.number = name
        self.config = config
        self.model_path = self.config.MODEL_PATH
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment_episodes = global_episodes.assign_add(1)
        self.global_steps = global_steps
        self.local_steps = 0
        self.increment_steps = global_steps.assign_add(1)
        self.cumulative_steps = logger_steps
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.train.SummaryWriter('train' + str(self.number), graph=tf.get_default_graph())
        self.env = Commander(self.number, self.config, self.name)   # RL training (the 'game')
        self.local_AC = ACNetwork(self.name, trainer, self.config)
        self.update_local_ops = update_target_graph('global', self.name)
        self.actions = self.env.action_space
        self.batch_rnn_state_init = None

        self.last_model_save_steps = 0
        self.last_log_writing_steps = 0

    def train(self, rollout, bootstrap_value, gamma, lmbda,  sess):

        """ Actor-Critic + Aux task training """

        # load an episode of experiences
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]
        if self.config.AUX_TASK_D2:
            aux_depth = rollout[:, 6]
        if self.config.GOAL_ON:
            goal_vector = np.vstack(gv for gv in rollout[:, -2])    # TODO: this implementation does not allow using goal vector without acceleration actions...
        if self.config.ACCELERATION_ACTIONS:
            velocity_state = np.vstack(vs for vs in rollout[:, -1])
        identity = np.eye(8)

        # calculating discounted return for each step
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(rewards_plus, gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]   # Advantage, approximated by TD error
        advantages = discount(advantages, lmbda)

        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(np.expand_dims(obs, 0) for obs in observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in: rnn_state
                     }

        ops_for_run = [self.local_AC.value_loss,
                       self.local_AC.policy_loss,
                       self.local_AC.entropy,
                       self.local_AC.grad_norms,
                       self.local_AC.var_norms,
                       self.local_AC.apply_grads]

        # if training with auxiliary tasks, augment feed_dict and loss ops
        if self.config.AUX_TASK_D2:
            depth_list = [np.vstack(identity[batch, :] for batch in px) for px in np.transpose(aux_depth)]
            depth_labels = np.swapaxes(np.array(depth_list), 0, 1)
            feed_dict.update({self.local_AC.aux_depth_labels[px]: depth_labels[px] for px in range(4*16)})
            ops_for_run.insert(3, self.local_AC.aux_depth2_loss)
        if self.config.GOAL_ON:
            feed_dict.update({self.local_AC.direction_input: goal_vector})
        if self.config.ACCELERATION_ACTIONS:
            feed_dict.update({self.local_AC.velocity_state: velocity_state})
        if self.config.PREV_ACTION_ON:
            prev_action = np.zeros((1, self.config.ACTIONS), dtype=np.float32)
            for i in range(len(actions)-1):
                prev_action = np.vstack((prev_action, np.eye(self.config.ACTIONS, dtype=np.float32)[actions[i]][:]))
            feed_dict.update({self.local_AC.prev_action: prev_action})
        if self.config.PREV_REWARD_ON:
            prev_reward = np.zeros((1, 1), dtype=float)
            for i in range(len(rewards) - 1):
                prev_reward = np.vstack((prev_reward, rewards[i]))
            feed_dict.update({self.local_AC.prev_reward: prev_reward})

        # calculate losses and gradients
        results = sess.run(ops_for_run, feed_dict=feed_dict)
        v_l, p_l, e_l = results[:3]
        g_n, v_n = results[-3:-1]
        self.batch_rnn_state_init = results[-1]

        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, sess, coord, saver):

        """ Target function for Thread.run(), policy evaluation on episodes + training """

        episode_count = sess.run(self.global_episodes)
        print('Starting worker ' + str(self.number))
        with sess.as_default(), sess.graph.as_default():

            # Training Loop
            while not coord.should_stop():
                # initializing episode
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                previous_reward = np.expand_dims(np.expand_dims(0.0, 0), 0)  # np.zeros([1, 1], dtype=float)
                previous_action = np.zeros([1, self.config.ACTIONS], dtype=np.float32)

                self.env.new_episode()
                s = self.env.get_observation()
                episode_frames.append(s)
                if self.config.AUX_TASK_D2:
                    aux_depth = np.expand_dims(self.env.get_observation(viewmode='depth').flatten(), 0)
                if self.config.GOAL_ON:
                    goal_direction = self.env.get_goal_direction()
                if self.config.ACCELERATION_ACTIONS:
                    velocity_state = self.env.get_velocity_state()
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state_init = rnn_state

                # Episode Loop
                while self.env.is_episode_finished() is False:

                    # running network for action selection
                    feed_dict = {self.local_AC.inputs: [s],
                                 self.local_AC.state_in: rnn_state}
                    if self.config.GOAL_ON:
                        feed_dict.update({self.local_AC.direction_input: goal_direction})
                    if self.config.ACCELERATION_ACTIONS:
                        feed_dict.update({self.local_AC.velocity_state: velocity_state})
                    if self.config.PREV_ACTION_ON:
                        feed_dict.update({self.local_AC.prev_action: previous_action})
                    if self.config.PREV_REWARD_ON:
                        feed_dict.update({self.local_AC.prev_reward: previous_reward})

                    a_dist, v, rnn_state = sess.run([self.local_AC.policy,
                                                     self.local_AC.value,
                                                     self.local_AC.state_out],
                                                    feed_dict=feed_dict)
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(np.equal(a_dist, a))
                    previous_action = np.expand_dims(np.eye(self.config.ACTIONS)[a][:], 0)  # onehot previous action

                    self.cumulative_steps.increment()
                    sess.run(self.increment_steps)
                    self.local_steps += 1

                    # Act and receive reward from th environment
                    r = self.env.action(self.actions[a])
                    d = self.env.is_episode_finished()
                    previous_reward = np.expand_dims(np.expand_dims(r, 0), 0)

                    if d is False:
                        s1 = self.env.get_observation()
                        episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_experiences = [s, a, r, s1, d, v[0, 0]]
                    if self.config.AUX_TASK_D2:
                        aux_depth = np.expand_dims(self.env.get_observation(viewmode='depth').flatten(), 0)
                        episode_experiences.append(aux_depth)
                    if self.config.GOAL_ON:
                        goal_direction = self.env.get_goal_direction()
                        episode_experiences.append(goal_direction)
                    if self.config.ACCELERATION_ACTIONS:
                        velocity_state = self.env.get_velocity_state()
                        episode_experiences.append(velocity_state)
                    episode_buffer.append(episode_experiences)
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    episode_step_count += 1

                    # running training step at the end of episode
                    if (len(episode_buffer) == self.config.STEPS_FOR_UPDATE) or d or\
                            (episode_step_count == self.config.MAX_EPISODE_LENGTH):
                        if d:
                            v1 = 0.0  # episode finished, no bootstrapping needed
                        else:
                            # bootstrap value from the last step for return calculation
                            feed_dict_v = {self.local_AC.inputs: [s],
                                           self.local_AC.state_in: rnn_state}
                            if self.config.GOAL_ON:
                                feed_dict_v.update({self.local_AC.direction_input: goal_direction})
                            if self.config.ACCELERATION_ACTIONS:
                                feed_dict_v.update({self.local_AC.velocity_state: velocity_state})
                            if self.config.PREV_ACTION_ON:
                                feed_dict_v.update({self.local_AC.prev_action: previous_action})
                            if self.config.PREV_REWARD_ON:
                                feed_dict_v.update({self.local_AC.prev_reward: previous_reward})
                            v1 = sess.run(self.local_AC.value,
                                          feed_dict=feed_dict_v)
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, v1, self.config.GAMMA, self.config.LAMBDA, sess)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d or (episode_step_count == self.config.MAX_EPISODE_LENGTH):
                        break

                # Summary writing, model saving, etc.
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                steps_since_log = self.local_steps - self.last_log_writing_steps

                if steps_since_log > self.config.LOGGING_PERIOD:
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

                    self.last_log_writing_steps = self.local_steps

                current_global_steps = sess.run(self.global_steps)
                if self.name == 'worker_0':
                    steps_since_save = current_global_steps - self.last_model_save_steps
                    if steps_since_save > self.config.MODEL_SAVE_PERIOD:
                        print('--- Saving model at {} global steps'.format(current_global_steps))
                        self.last_model_save_steps = current_global_steps -\
                            (current_global_steps % self.config.MODEL_SAVE_PERIOD)
                        saver.save(sess, self.model_path + '/model-' + str(int(self.last_model_save_steps/1000)) + 'k.cptk')
                    sess.run(self.increment_episodes)
                    print('--- worker_0 {} episodes, {} global steps'.format(episode_count, current_global_steps))
                    if current_global_steps > self.config.MAX_STEPS:
                        coord.request_stop()
                episode_count += 1

            # shutting down client and sim
            print('Shutting down {}...    '.format(self.name))
            self.env.shut_down()
            print('       Sim and client closed.')


class Player:

    """ A3C Agent for evaluation. """

    def __init__(self, number, config):
        self.name = 'player_' + str(number)
        print('Initializing {} ...'.format(self.name))
        self.config = config
        self.number = number
        self.local_AC = ACNetwork('player_{}'.format(self.number), None, self.config)
        self.update_local_ops = update_target_graph('global', 'player_{}'.format(self.number))
        self.env = Commander(self.number, self.config, self.name)
        self.actions = self.env.action_space
        self.episodes_started = 0
        self.episodes_finished = 0
        self.steps = 0
        self.rnn_state = None
        self.s = None
        self.stop_requested = False
        self.crashes = 0
        self.terminations = 0

        # remove previous trajectory logs to avoid unintentional appending
        ucv_utils.remove_file('./trajectory_{}.yaml'.format(self.name))

        print('[{}] initialization done.'.format(self.name))

    def play(self, session, coordinator):

        with session.as_default(), session.graph.as_default():
            session.run(self.update_local_ops)  # loading weights

            # evaluation loop
            while not coordinator.should_stop():

                finished_episode = False
                self.s = self.env.get_observation()
                self.rnn_state = self.local_AC.state_init
                self.steps = 0
                self.episodes_started += 1
                previous_reward = np.expand_dims(np.expand_dims(0.0, 0), 0)  # np.zeros([1, 1], dtype=float)
                previous_action = np.zeros([1, self.config.ACTIONS], dtype=np.float32)
                if self.config.ACCELERATION_ACTIONS:
                    velocity_state = self.env.get_velocity_state()

                # episode loop
                while not finished_episode:
                    feed_dict = {self.local_AC.inputs: [self.env.get_observation()],
                                 self.local_AC.state_in: self.local_AC.state_init}
                    if self.config.GOAL_ON:
                        goal_direction = self.env.get_goal_direction()
                        feed_dict.update({self.local_AC.direction_input: goal_direction})
                    if self.config.PREV_ACTION_ON:
                        feed_dict.update({self.local_AC.prev_action: previous_action})
                    if self.config.PREV_REWARD_ON:
                        feed_dict.update({self.local_AC.prev_reward: previous_reward})
                    if self.config.ACCELERATION_ACTIONS:
                        feed_dict.update({self.local_AC.velocity_state: velocity_state})

                    if self.config.AUX_TASK_D2:
                        # no training, no need for depth images
                        # self.env.get_observation(viewmode='depth')
                        pass

                    # running inference on network, action selection
                    a_dist, self.rnn_state = session.run([self.local_AC.policy, self.local_AC.state_out], feed_dict=feed_dict)
                    if self.config.STOCHASTIC_POLICY_EVAL:
                        a = np.random.choice(a_dist[0], p=a_dist[0])
                        a = np.argmax(np.equal(a_dist, a))
                    else:
                        a = np.argmax(a_dist)

                    reward = self.env.action(self.actions[a])
                    self.steps += 1
                    previous_action = np.expand_dims(np.eye(self.config.ACTIONS)[a][:], 0)  # onehot previous action
                    previous_reward = np.expand_dims(np.expand_dims(reward, 0), 0)

                    if self.steps > self.config.MAX_EVALUATION_EPISODE_LENGTH:
                        self.env.new_episode(save_trajectory=True)
                        self.terminations += 1
                        finished_episode = True

                    if self.env.is_episode_finished():
                        self.env.new_episode(save_trajectory=True)
                        self.crashes += 1
                        finished_episode = True
                self.episodes_finished += 1

            self.env.shut_down()






