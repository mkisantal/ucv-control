import unrealcv
from manual_ctrl import ManualController
import ucv_utils
from pynput.keyboard import Listener
from command import Commander
import network as net
import tensorflow as tf
import os
import threading
from time import sleep
import subprocess

# setup
ManualControlEnabled = True
(HOST, PORT) = ('localhost', 9000)
sim_dir = '/home/mate/Documents/ucv-pkg/LinuxNoEditor/unrealCVfirst/Binaries/Linux/'


if ManualControlEnabled:
    ucv_utils.set_port(PORT, sim_dir)
    print('Starting simulator instance.')
    subprocess.Popen(sim_dir + 'unrealCVfirst-Linux-Shipping')
    sleep(5)
    print('Client connecting...')
    client = unrealcv.Client((HOST, PORT))
    client.connect()

    cmd = Commander(client, goal_heading_deg=90)
    manual = ManualController(cmd)

    with Listener(on_press=manual.on_press, on_release=manual.on_release) as listener:
        listener.join()

else:
    # do RL training
    max_episode_length = 300
    gamma = 0.99
    s_size = 7056
    a_size = 4
    load_model = False
    model_path = './model'

    tf.reset_default_graph()

    num_workers = 4
    cmd = []
    clients = []
    # start simulation instances
    for i in range(num_workers):

        ucv_utils.set_port(PORT+i, sim_dir)

        print('Starting simulator instance {}'.format(i))
        subprocess.Popen(sim_dir + 'unrealCVfirst-Linux-Shipping')
        sleep(5)

        print('connecting client instance...')
        clients.append(unrealcv.Client((HOST, PORT+i)))
        clients[i].connect()

        cmd.append(Commander(clients[i], goal_heading_deg=90))

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = net.ACNetwork(len(cmd[0].action_space), cmd[0].state_space_size, 'global', None)

        # num_workers = multiprocessing.cpu_count()

        workers = []
        for i in range(num_workers):
            workers.append(net.Worker(i, model_path, trainer, global_episodes, cmd[i]))
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
        coord.join(worker_threads)
