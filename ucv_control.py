# from manual_ctrl import ManualController
# from pynput.keyboard import Listener
import network as net
import tensorflow as tf
import os
import threading
from time import sleep


# setup
ManualControlEnabled = False
(HOST, PORT) = ('localhost', 9000)
sim_dir = '/home/mate/Documents/ucv-pkg2/LinuxNoEditor/unrealCVfirst/Binaries/Linux/'


if ManualControlEnabled:
    #  ucv_utils.set_port(PORT, sim_dir)
    # print('Starting simulator instance.')
    # client = unrealcv.Client((HOST, PORT))
    # sim = ucv_utils.start_sim(sim_dir, client)
    # cmd = Commander(client, sim_dir, sim)
    # manual = ManualController(cmd)
    #
    # with Listener(on_press=manual.on_press, on_release=manual.on_release) as listener:
    #     listener.join()
    print('Manual control is not supported on server.')

else:
    # do RL training
    max_episode_length = 300
    gamma = 0.99
    s_size = 7056
    s_shape = [84, 84, 3]  # for now RGB
    a_size = 3
    load_model = False
    model_path = './model'

    tf.reset_default_graph()

    num_workers = 8

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = net.ACNetwork(a_size, s_shape, 'global', None)

        # num_workers = multiprocessing.cpu_count()

        workers = []
        for i in range(num_workers):
            workers.append(net.Worker(i, model_path, trainer, global_episodes))
        saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
            t = threading.Thread(target=worker_work)
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        while not coord.should_stop():
            try:
                sleep(0.1)
            except KeyboardInterrupt:
                print('terminating threads.....')
                coord.request_stop()
        coord.join(worker_threads)
    print('Tot ziens')
