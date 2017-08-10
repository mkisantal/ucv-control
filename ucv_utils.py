import fileinput
import re
import subprocess
from time import sleep
from random import randint


def set_port(port, sim_dir):
    for line in fileinput.FileInput(sim_dir + 'unrealcv.ini', inplace=1):
        if "Port=" in line:
            line = re.sub('\d{4}', str(port), line)
        print line,


def start_sim(sim_dir, client):
    got_connection = False
    attempt = 1
    while not got_connection:
        if attempt > 2:
            wait_time = 20 + randint(5, 20)  # rand to avoid too many parallel sim startups
            print('Multiple start attempts failed. Trying again in {} seconds.'.format(wait_time))
            sleep(wait_time)
            attempt = 1
        print('Connection attempt: {}'.format(attempt))
        attempt += 1
        sim = subprocess.Popen(sim_dir + 'unrealCVfirst-Linux-Shipping')
        sleep(5)
        client.connect()
        sleep(2)
        got_connection = client.isconnected()
        if not got_connection:
            sim.terminate()
            sleep(3)
        else:
            return sim
