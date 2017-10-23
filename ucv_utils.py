import fileinput
import re
import os
from subprocess import Popen
from config import Config
from time import sleep


def set_port(port, sim_dir):
    try:
        for line in fileinput.FileInput(sim_dir + 'unrealcv.ini', inplace=1):
            if "Port=" in line:
                line = re.sub('\d{4}', str(port), line)
            if "Width=" in line:
                line = re.sub('\d{2,4}', str(84), line)
            if "Height=" in line:
                line = re.sub('\d{2,4}', str(84), line)
            print line,
    except OSError or IOError:
        print('unrealcv.ini does not exist, launching Sim to create it')
        with open(os.devnull, 'w') as fp:
            sim = Popen(Config.SIM_DIR + Config.SIM_NAME, stdout=fp)
        sleep(5)
        sim.terminate()
        set_port(port, sim_dir)
