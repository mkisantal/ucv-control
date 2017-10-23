import network as net
import tensorflow as tf
import os
import threading
from time import sleep
from logger import TestLogger, CumulativeStepsLogger
from config import Config

model_path = Config.MODEL_PATH
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):

    # initializing a master network
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = net.ACNetwork('global', None)

    # initializing workers
    workers = []
    cumulative_steps = CumulativeStepsLogger()
    for i in range(Config.NUM_WORKERS):
        workers.append(net.Worker(i, trainer, global_episodes, cumulative_steps))
    saver = tf.train.Saver()

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    # weight initialization
    if Config.LOAD_MODEL:
        print('Loading model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

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
print('Tot ziens')
