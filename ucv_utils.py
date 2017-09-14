import fileinput
import re

def set_port(port, sim_dir):
    for line in fileinput.FileInput(sim_dir + 'unrealcv.ini', inplace=1):
        if "Port=" in line:
            line = re.sub('\d{4}', str(port), line)
        if "Width=" in line:
            line = re.sub('\d{2,4}', str(84), line)
        if "Height=" in line:
            line = re.sub('\d{2,4}', str(84), line)
        print line,
