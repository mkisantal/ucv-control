import network as net
import tensorflow as tf
import os
import threading
from time import sleep
from logger import TestLogger, CumulativeStepsLogger
from config import Config
from integrated_evaluation import coordinate_evaluation

model_path = Config.MODEL_PATH
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):

    # initializing a master network
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    global_steps = tf.Variable(0, dtype=tf.int32, name='global_steps', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = net.ACNetwork('global', None)

    # initializing workers
    workers = []
    logger_steps = CumulativeStepsLogger()
    for i in range(Config.NUM_WORKERS):
        workers.append(net.Worker(i, trainer, global_episodes, global_steps, logger_steps))
    saver = tf.train.Saver(max_to_keep=1000)
    gpu_options = tf.GPUOptions(visible_device_list='0')

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    coord = tf.train.Coordinator()

    # weight initialization
    if Config.LOAD_MODEL:
        print('Loading model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt != None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())

    # starting logger threads
    logger = TestLogger(workers)
    threading.Thread(target=lambda: logger.work()).start()
    threading.Thread(target=lambda: logger_steps.work()).start()
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

            # monitoring threads
            for i in range(len(worker_threads)):
                if not worker_threads[i].is_alive():
                    print('Thread {} is dead. Restart attempt...'.format(i))
                    worker_work = lambda: workers[i].work(sess, coord, saver)
                    worker_threads[i] = threading.Thread(target=worker_work)
                    worker_threads[i].start()
                if workers[0].start_eval:
                    coordinate_evaluation(workers, coord)

        except KeyboardInterrupt:
            print('terminating threads.....')
            coord.request_stop()

    # stop training
    logger.should_stop = True
    logger_steps.should_stop = True
    coord.join(worker_threads)
print('Tot ziens')
