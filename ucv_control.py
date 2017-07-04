from __future__ import print_function
import unrealcv
import math
import numpy as np
# manual control stuff
from pynput.keyboard import Key, Listener, KeyCode

trajectory = []
goal_heading = 90
goal_vector = [math.cos(math.radians(goal_heading)), math.sin(math.radians(goal_heading)), 0.0]

(HOST, PORT) = ('localhost', 9000)
client = unrealcv.Client((HOST, PORT))


def calculate_reward(goal, displacement, collision=False):
    goal_direction_reward = 1.0
    crash_reward = -10.0

    norm_displacement = np.array(displacement) / np.linalg.norm(np.array(displacement))
    reward = np.dot(np.array(goal), norm_displacement) * goal_direction_reward
    if collision:
        reward += crash_reward

    print('reward: {}'.format(reward))

    return reward


def get_pos(print_pos=False):

    if len(trajectory) == 0:
        rot = [float(v) for v in client.request('vget /camera/0/rotation').split(' ')]
        loc = [float(v) for v in client.request('vget /camera/0/location').split(' ')]
        trajectory.append(dict(location=loc, rotation=rot))
    else:
        loc = trajectory[-1]["location"]
        rot = trajectory[-1]["rotation"]

    if print_pos:
        print('Position x={} y={} z={}'.format(*loc))
        print('Rotation pitch={} heading={} roll={}'.format(*rot))

    return loc, rot


def move(loc_cmd=0.0, rot_cmd=(0.0, 0.0, 0.0)):
    loc, rot = get_pos()
    new_rot = [sum(x) for x in zip(rot, rot_cmd)]
    displacement = [loc_cmd*math.cos(math.radians(rot[1])), loc_cmd*math.sin(math.radians(rot[1])), 0.0]
    new_loc = [sum(x) for x in zip(loc, displacement)]
    collision = False

    if rot_cmd != (0.0, 0.0, 0.0):
        res = client.request('vset /camera/0/rotation {} {} {}'.format(*new_rot))
        assert res == 'ok', 'Fail to set camera rotation'
    if loc_cmd != 0.0:
        res = client.request('vset /camera/0/moveto {} {} {}'.format(*new_loc))
        print(res)
        if res != 'ok':
            print('Collision. Failed to move to position.')
            collision = True
            new_loc = [float(v) for v in res.split(' ')]

    trajectory.append(dict(location=new_loc, rotation=new_rot))

    calculate_reward(goal_vector, displacement=displacement, collision=collision)
    return


def velocity_command(cmd):
    angle = 20.0  # degrees/step
    speed = 20.0  # cm/step

    if cmd == 'left':
        move(loc_cmd=speed, rot_cmd=[0, -angle, 0])
    elif cmd == 'right':
        move(loc_cmd=speed, rot_cmd=[0, angle, 0])
    elif cmd == 'forward':
        move(loc_cmd=speed)
    elif cmd == 'backward':
        move(loc_cmd=-speed)


def save_view():
    res = client.request('vget /viewmode')
    res2 = client.request('vget /camera/0/' + res)
    print(res2)
    return


def change_view():
    switch = dict(lit='normal', normal='depth', depth='object_mask', object_mask='lit')
    res = client.request('vget /viewmode')
    res2 = client.request('vset /viewmode ' + switch[res])
    print(res2)
    return


def on_press(key):
    if key == KeyCode.from_char("p"):
        print('a key pressed, getting position')
        get_pos(True)
    if key == KeyCode.from_char("w"):
        velocity_command('forward')
    elif key == KeyCode.from_char("s"):
        velocity_command('backward')
    elif key == KeyCode.from_char("a"):
        velocity_command('left')
    elif key == KeyCode.from_char("d"):
        velocity_command('right')

    if key == Key.space:
        save_view()

    if key == Key.tab:
        change_view()


def on_release(key):
    if key == Key.esc:
        # Stop listener
        return False

client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')


with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()





