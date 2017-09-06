# action
import math
import numpy as np
import StringIO
import PIL.Image
from random import randint
from time import sleep
import subprocess
import ucv_utils
import exceptions


class Commander:

    def __init__(self, client, sim_dir, sim):
        self.client = client
        self.trajectory = []

        # navigation goal direction
        self.goal_heading = 0
        self.goal_vector = [math.cos(math.radians(self.goal_heading)), math.sin(math.radians(self.goal_heading)), 0.0]

        # RL rewards
        self.goal_direction_reward = 1.0
        self.crash_reward = -10.0

        # Agent actions
        self.action_space = ('left', 'right', 'forward')  #  'backward'
        self.state_space_size = [84, 84, 3]  # for now RGB

        self.episode_finished = False
        self.should_terminate = False

    def action(self, cmd):
        angle = 20.0  # degrees/step
        speed = 20.0  # cm/step
        loc_cmd = [0.0, 0.0, 0.0]
        rot_cmd = [0.0, 0.0, 0.0]
        if cmd == 'left':
            # move(loc_cmd=speed, rot_cmd=[0, -angle, 0])
            loc_cmd[0] = speed
            rot_cmd[1] = -angle
        elif cmd == 'right':
            # move(loc_cmd=speed, rot_cmd=[0, angle, 0])
            loc_cmd[0] = speed
            rot_cmd[1] = angle
        elif cmd == 'forward':
            # move(loc_cmd=speed)
            loc_cmd[0] = speed
        elif cmd == 'backward':
            # move(loc_cmd=-speed)
            loc_cmd[0] = -speed

        reward = self.move(loc_cmd[0], rot_cmd)  # TODO: change this to full loc_cmd vector
        return reward

    def sim_command(self, cmd):
        if cmd == 'save_view':
            self.save_view()
        elif cmd == 'change_view':
            self.change_view()
        elif cmd == 'get_position':
            self.get_pos(print_pos=True)
        return

    def save_view(self):
        res = self.client.request('vget /viewmode')
        res2 = self.client.request('vget /camera/0/' + res)
        print(res2)
        return

    def change_view(self, viewmode=''):
        if viewmode == '':
            switch = dict(lit='normal', normal='depth', depth='object_mask', object_mask='lit')
            res = self.client.request('vget /viewmode')
            res2 = self.client.request('vset /viewmode ' + switch[res])
            # print(res2)
        elif viewmode in {'lit', 'normal', 'depth', 'object_mask'}:
            res2 = self.client.request('vset /viewmode ' + viewmode)
        return

    def get_pos(self, print_pos=False):

        if len(self.trajectory) == 0:
            rot = [float(v) for v in self.client.request('vget /camera/0/rotation').split(' ')]
            loc = [float(v) for v in self.client.request('vget /camera/0/location').split(' ')]
            self.trajectory.append(dict(location=loc, rotation=rot))
        else:
            loc = self.trajectory[-1]["location"]
            rot = self.trajectory[-1]["rotation"]

        if print_pos:
            print('Position x={} y={} z={}'.format(*loc))
            print('Rotation pitch={} heading={} roll={}'.format(*rot))

        return loc, rot

    def reset_agent(self):
        new_loc = self.trajectory[-1]["location"]
        new_rot = self.trajectory[-1]["rotation"]
        res1 = self.client.request('vset /camera/0/rotation {:.3f} {:.3f} {:.3f}'.format(*new_rot))
        res2 = self.client.request('vset /camera/0/moveto {:.2f} {:.2f} {:.2f}'.format(*new_loc))

        if not res1 or not res2:

            raise TypeError

        return

    def move(self, loc_cmd=0.0, rot_cmd=(0.0, 0.0, 0.0)):
        loc, rot = self.get_pos()
        new_rot = [sum(x) for x in zip(rot, rot_cmd)]
        displacement = [loc_cmd * math.cos(math.radians(rot[1])), loc_cmd * math.sin(math.radians(rot[1])), 0.0]
        new_loc = [sum(x) for x in zip(loc, displacement)]
        collision = False

        if rot_cmd != (0.0, 0.0, 0.0):
            res = self.client.request('vset /camera/0/rotation {:.3f} {:.3f} {:.3f}'.format(*new_rot))
            if res != 'ok':
                if res is None:
                    print ('Simulator restart required.')
                    raise TypeError
        if loc_cmd != 0.0:
            res = self.client.request('vset /camera/0/moveto {:.2f} {:.2f} {:.2f}'.format(*new_loc))
            if res != 'ok':
                if res is None:
                    print ('Simulator restart required.')
                    raise TypeError
                else:
                    collision = True
                    new_loc = [float(v) for v in res.split(' ')]

        self.trajectory.append(dict(location=new_loc, rotation=new_rot))

        reward = self.calculate_reward(displacement=displacement, collision=collision)
        if collision:
            self.episode_finished = True
        return reward

    def calculate_reward(self, displacement, collision=False):
        reward = 0
        distance = np.linalg.norm(np.array(displacement))
        if distance != 0:
            norm_displacement = np.array(displacement) / distance
            reward += np.dot(np.array(self.goal_vector), norm_displacement) * self.goal_direction_reward
        if collision:
            reward += self.crash_reward

        # print('reward: {}'.format(reward))

        return reward

    @staticmethod
    def _read_npy(res):
        return np.load(StringIO.StringIO(res))

    @staticmethod
    def _read_png(res):
        img = PIL.Image.open(StringIO.StringIO(res))
        return np.asarray(img)

    def get_observation(self, grayscale=False, show=False):
        res = self.client.request('vget /camera/0/lit png')
        rgba = self._read_png(res)
        rgb = rgba[:, :, :3]
        if grayscale is True:
            observation = np.mean(rgb, 2)
        else:
            observation = rgb

        if show:
            img = PIL.Image.fromarray(observation)
            img.show()

        return observation

    def new_episode(self):
        # simple respawn: just turn around 180+/-60 deg
        self.move(rot_cmd=(0.0, randint(120, 240), 0.0))
        self.goal_heading = randint(0, 360)
        self.episode_finished = False
        return

    def is_episode_finished(self):
        return self.episode_finished

