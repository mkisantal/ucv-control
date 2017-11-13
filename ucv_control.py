import network as net
import tensorflow as tf
import os
import threading
from time import sleep
from logger import TestLogger, CumulativeStepsLogger
from config import Configuration
from tensorflow.contrib import slim

def main(mode, episodes):
    config = Configuration(mode, episodes)
    model_path = config.MODEL_PATH
    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with tf.device("/cpu:0"):

        # initializing a master network
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = net.ACNetwork('global', None, config)
        var_to_restore = slim.get_variables_to_restore()  # restore only master network
        saver = tf.train.Saver(var_to_restore)

        if config.TRAIN_MODE:
            # initializing workers for training
            workers = []
            cumulative_steps = CumulativeStepsLogger()
            for i in range(config.NUM_WORKERS):
                workers.append(net.Worker(i, trainer, global_episodes, cumulative_steps, config))

        else:
            # initializing players for evaluation
            players = []
            for i in range(config.NUM_WORKERS):
                players.append(net.Player(i, config))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        # weight initialization
        if config.LOAD_MODEL or not config.TRAIN_MODE:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        if config.TRAIN_MODE:
            # starting logger threads
            logger = TestLogger(workers)
            threading.Thread(target=lambda: logger.work()).start()
            threading.Thread(target=lambda: cumulative_steps.work()).start()
            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(sess, coord, saver)
                t = threading.Thread(target=worker_work)
                t.start()
                sleep(0.5)
                worker_threads.append(t)

            # keep threads running
            while not coord.should_stop():
                try:
                    sleep(0.1)
                    for i in range(len(worker_threads)):
                        if not worker_threads[i].is_alive():
                            print('Thread {} is dead. Restart attempt...'.format(i))
                            worker_work = lambda: workers[i].work(sess, coord, saver)
                            worker_threads[i] = threading.Thread(target=worker_work)
                            worker_threads[i].start()
                except KeyboardInterrupt:
                    print('terminating threads.....')
                    coord.request_stop()

            # stop training
            logger.should_stop = True
            cumulative_steps.should_stop = True
            coord.join(worker_threads)
        else:
            player_threads = []
            for player in players:
                player_play = lambda: player.play(sess, coord)
                t = threading.Thread(target=player_play)
                t.start()
                sleep(0.5)
                player_threads.append(t)

            eval_finished = False
            finished_episodes = 0
            while not eval_finished:
                try:
                    sleep(0.1)
                    # check episodes
                    episode_count = 0
                    for player in players:
                        episode_count += player.episode_count
                    if episode_count > finished_episodes:
                        finished_episodes = episode_count
                        print('Finished {} episodes.'.format(finished_episodes))
                    if episode_count > config.MAX_EPISODES_FOR_EVAL-1:
                        # terminate threads
                        print('Terminating evaluation after {} finished episodes.'.format(finished_episodes))
                        # for player in players:
                        #     player.should_stop = True
                        coord.request_stop()
                        eval_finished = True

                except KeyboardInterrupt:
                    print('terminating threads.....')
                    # for player in players:
                    #     player.should_stop = True
                    coord.request_stop()
            coord.join(player_threads)
            for player in players:
                player.env.shut_down()

    print('Tot ziens')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', help='Set either \'train\' or \'eval\'', default='eval')
    parser.add_argument('--episodes', help='Number of episodes for running training or evaluation', default=0)
    args = parser.parse_args()

    main(mode=args.mode, episodes=args.episodes)
