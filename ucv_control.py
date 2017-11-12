import network as net
import tensorflow as tf
import os
import threading
from time import sleep
from logger import TestLogger, CumulativeStepsLogger
from config import Config
from tensorflow.contrib import slim

model_path = Config.MODEL_PATH
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):

    # initializing a master network
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = net.ACNetwork('global', None)
    var_to_restore = slim.get_variables_to_restore()  # restore only master network
    saver = tf.train.Saver(var_to_restore)

    if Config.TRAIN_MODE:
        # initializing workers for training
        workers = []
        cumulative_steps = CumulativeStepsLogger()
        for i in range(Config.NUM_WORKERS):
            workers.append(net.Worker(i, trainer, global_episodes, cumulative_steps))

    else:
        # initializing players for evaluation
        players = []
        for i in range(Config.NUM_WORKERS):
            players.append(net.Player(i))


with tf.Session() as sess:
    coord = tf.train.Coordinator()

    # weight initialization
    if Config.LOAD_MODEL or not Config.TRAIN_MODE:
        print('Loading model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    if Config.TRAIN_MODE:
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
                if episode_count > Config.MAX_EPISODES_FOR_EVAL-1:
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
