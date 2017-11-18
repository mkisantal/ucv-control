from __future__ import print_function
import tensorflow as tf
from subprocess import Popen
from time import sleep
import os
import errno
from config import Configuration
import re


def set_port(port, sim_dir, config):
    try:
        with open(sim_dir + 'unrealcv.ini', 'w') as ini_file:
            print('[UnrealCV.Core]', file=ini_file)
            print('Port={}'.format(str(port)), file=ini_file)
            print('Width=84', file=ini_file)
            print('Height=84', file=ini_file)
    except (OSError, IOError) as err:
        print(err)
        print('unrealcv.ini does not exist, launching Sim to create it')
        with open(os.devnull, 'w') as fp:
            sim = Popen(config.SIM_DIR + config.SIM_NAME, stdout=fp)
        sleep(5)
        sim.terminate()
        set_port(port, sim_dir, config)


def remove_file(fname):

    """ Remove file, if it exist. """
    
    try:
        os.remove(fname)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def print_checkpoint_steps():
    config = Configuration('eval', 0)
    ckpt = tf.train.get_checkpoint_state(config.MODEL_PATH)
    if ckpt is None:
        print(0)
    else:
        model_name = re.search('model-.*.cptk', ckpt.model_checkpoint_path).group(0)[6:-6]
        print(int(model_name) * 1000)

if __name__ == '__main__':
    print_checkpoint_steps()