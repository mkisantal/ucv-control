import tensorflow as tf
import numpy as np
import network as net
from command import Commander



class Player:
    def __init__(self, env):
        self.network = net.ACNetwork(len(env.action_space), env.state_space_size, 'global', None)
        self.env = env
        self.s = None
        self.rnn_state = None
        self.actions = self.env.action_space

    def play(self, session):

        if self.s is None or self.rnn_state is None:    # first step
            self.s = self.env.get_observation()
            self.rnn_state = self.network.state_init

        feed_dict = {self.network.inputs: [self.env.get_observation()],
                     self.network.state_in[0]: self.network.state_init[0],
                     self.network.state_in[1]: self.network.state_init[1]}

        a_dist, self.rnn_state = session.run([self.network.policy, self.network.state_out], feed_dict=feed_dict)
        a = np.argmax(a_dist)

        reward = self.env.action(self.actions[a])
        if reward == -10:
            print('Crash.')
            self.env.new_episode()
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
        while True:
            try:
                player.play(sess)
            except KeyboardInterrupt:
                print('\nShutting down...')
                cmd.shut_down()
                break
