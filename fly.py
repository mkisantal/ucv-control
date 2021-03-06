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
            self.rnn_state = self.network.state_init

        feed_dict = {self.network.inputs: [self.env.get_observation()],
                     self.network.state_in[0]: self.network.state_init[0],
                     self.network.state_in[1]: self.network.state_init[1]}
        if Config.GOAL_ON:
            goal_direction = self.env.get_goal_direction()
            feed_dict.update({self.network.direction_input: goal_direction})

        if Config.AUX_TASK_D2:
            cmd.get_observation(viewmode='depth')
            pass

        a_dist, self.rnn_state = session.run([self.network.policy, self.network.state_out], feed_dict=feed_dict)
        a = np.argmax(a_dist)

        reward = self.env.action(self.actions[a])
        self.steps += 1
        if self.steps > Config.MAX_EVALUATION_EPISODE_LENGTH:
            print('Max episode length reached. Resetting agent.')
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

    if not Config.EVAL_MODE:
        print('\nNot in evaluation mode.\nPlease EVAL_MODE in Config to \'True\'! Shutting down...')
        exit()

    cmd = Commander(0)

    tf.reset_default_graph()
    player = Player(cmd)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        model_path = './model'
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        while player.episode_count < 20:
            try:
                player.play(sess)
            except KeyboardInterrupt:
                print('\nShutting down...')
                cmd.shut_down()
                break
        print('Evaluation for {} episodes done.'.format(player.episode_count))
        cmd.shut_down()
