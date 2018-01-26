import tensorflow as tf
import numpy as np
import network as net
from command import Commander
from config import Config


class Player:
    def __init__(self, env):
        self.network = net.ACNetwork('global', None)
        self.env = env
        self.env.new_episode(save_trajectory=False)
        self.s = None
        self.rnn_state = None
        self.actions = self.env.action_space
        self.steps = 0
        self.episode_count = 0

    def play(self, session):

        if self.s is None or self.rnn_state is None:    # first step
            self.s = self.env.get_observation()
            if Config.USE_LSTM:
                self.rnn_state = self.network.state_init

        feed_dict = {self.network.inputs: [self.env.get_observation()]}
        ops_to_run = [self.network.policy]
        if Config.USE_LSTM:
            feed_dict.update({self.network.state_in[0]: self.rnn_state[0],
                              self.network.state_in[1]: self.rnn_state[1]})
            ops_to_run.append(self.network.state_out)
        if Config.GOAL_ON:
            goal_direction = self.env.get_goal_direction()
            feed_dict.update({self.network.direction_input: goal_direction})

        if Config.AUX_TASK_D2:
            cmd.get_observation(viewmode='depth')
            pass

        if Config.USE_LSTM:
            a_dist, self.rnn_state = session.run(ops_to_run, feed_dict=feed_dict)
        else:
            a_dist = session.run(ops_to_run, feed_dict=feed_dict)
        a = np.argmax(a_dist)

        reward = self.env.action(self.actions[a])
        self.steps += 1
        print(self.steps)
        if self.steps > 800:
            print('Reset.')
            self.steps = 0
            self.episode_count += 1
            print(self.episode_count)
            self.env.new_episode(save_trajectory=True)
            self.rnn_state = None
            self.s = None
        print('Reward: {}'.format(reward))
        if cmd.is_episode_finished():
            print('Collision.')
            self.episode_count += 1
            print(self.episode_count)
            self.env.new_episode(save_trajectory=True)
            self.steps = 0
            self.rnn_state = None
            self.s = None
            self.s = self.env.get_observation()

        return


if __name__ == '__main__':

    cmd = Commander(0)

    tf.reset_default_graph()
    player = Player(cmd)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        model_path = './model'
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        while player.episode_count < 12:
            try:
                player.play(sess)
            except KeyboardInterrupt:
                print('\nShutting down...')
                cmd.shut_down()
                break
        print('Evaluation for {} episodes done.'.format(player.episode_count))
