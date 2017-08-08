import fileinput
import re


def set_port(port, sim_dir):
    for line in fileinput.FileInput(sim_dir + 'unrealcv.ini', inplace=1):
        if "Port=" in line:
            line = re.sub('\d{4}', str(port), line)
        print line,
