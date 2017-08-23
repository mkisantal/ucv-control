import fileinput
import re
import subprocess
from time import sleep
from random import randint
import os


def set_port(port, sim_dir):
    for line in fileinput.FileInput(sim_dir + 'unrealcv.ini', inplace=1):
        if "Port=" in line:
            line = re.sub('\d{4}', str(port), line)
        if "Width=" in line:
            line = re.sub('\d{3}', str(84), line)
        if "Height=" in line:
            line = re.sub('\d{3}', str(84), line)
        print line,


def start_sim(sim_dir, client, cmd):
    got_connection = False
    attempt = 1
    while not got_connection:
        # for i in range(10):
        if cmd is not None:
            if cmd.should_terminate:
                return
        if attempt > 2:
            wait_time = 20 + randint(5, 20)  # rand to avoid too many parallel sim startups
            print('Multiple start attempts failed. Trying again in {} seconds.'.format(wait_time))
            sleep(wait_time)
            attempt = 1
        print('Connection attempt: {}'.format(attempt))
        attempt += 1
        with open(os.devnull, 'w') as fp:
            sim = subprocess.Popen(sim_dir + 'RealisticRendering', stdout=fp)
        sleep(10)
        client.connect()
        sleep(2)
        got_connection = client.isconnected()
        if not got_connection:
            sim.terminate()
            sleep(3)
        else:
            return sim
