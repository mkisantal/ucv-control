from __future__ import print_function
import os
from subprocess import Popen
from time import sleep


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
        set_port(port, sim_dir)
