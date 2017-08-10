import fileinput
import re
import subprocess
from time import sleep


def set_port(port, sim_dir):
    for line in fileinput.FileInput(sim_dir + 'unrealcv.ini', inplace=1):
        if "Port=" in line:
            line = re.sub('\d{4}', str(port), line)
        print line,


def start_sim(sim_dir, client):
    got_connection = False

    while not got_connection:
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
