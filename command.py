# action
import math
import numpy as np
import StringIO
import PIL.Image
from random import randint, sample
import time
import subprocess
import ucv_utils
from unrealcv import Client
import os
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from config import Config
import yaml


class Commander:

    def __init__(self, number, mode=None):
        self.trajectory = []
        self.name = 'worker_' + str(number)

        # navigation goal direction
        self.goal_location = None
        self.goal_vector = None

        self.locations = []
        with open(Config.SIM_DIR + 'locations.yaml', 'r') as loc_file:
            self.locations = yaml.load(loc_file)

        # RL rewards
        self.goal_direction_reward = 1.0
        self.crash_reward = -10.0

        # Agent actions
        self.action_space = ('left', 'right', 'forward')  #  'backward'
        self.state_space_size = Config.STATE_SHAPE  # for now RGB
        self.speed = 20.0  # cm/step

        self.episode_finished = False
        self.should_terminate = False

        self.sim = None
        self.client = Client((Config.HOST, Config.PORT + number))
        self.should_stop = False
        self.mode = mode
        if self.mode == 'test':
            self.client.connect()
        else:
            self.start_sim()

    def shut_down(self):
        if self.client.isconnected():
            self.client.disconnect()
        if self.sim is not None:
            self.sim.terminate()
            self.sim = None

    def start_sim(self, restart=False):
        # disconnect and terminate if restarting
        attempt = 1
        got_connection = False
        while not got_connection and not self.should_stop:
            self.shut_down()
            port = self.client.message_client.endpoint[1]
            ucv_utils.set_port(port, Config.SIM_DIR)
            print('Connection attempt: {}'.format(attempt))
            with open(os.devnull, 'w') as fp:
                self.sim = subprocess.Popen(Config.SIM_DIR + Config.SIM_NAME, stdout=fp)
            attempt += 1
            time.sleep(10)
            self.client.connect()
            time.sleep(2)
            got_connection = self.client.isconnected()
            if got_connection:
                if restart:
                    try:
                        self.reset_agent()
                    except TypeError:
                        got_connection = False
            else:
                if attempt > 2:
                    wait_time = 20 + randint(5, 20)  # rand to avoid too many parallel sim startups
                    print('Multiple start attempts failed. Trying again in {} seconds.'.format(wait_time))
                    waited = 0
                    while not self.should_stop and (waited < wait_time):
                        time.sleep(1)
                        waited += 1
                    attempt = 1
        return

    def reconnect(self):
        print('{} trying to reconnect.'.format(self.name))
        self.client.disconnect()
        time.sleep(2)
        self.client.connect()
        return

    def action(self, cmd):
        angle = 20.0  # degrees/step
        loc_cmd = [0.0, 0.0, 0.0]
        rot_cmd = [0.0, 0.0, 0.0]
        if cmd == 'left':
            # move(loc_cmd=speed, rot_cmd=[0, -angle, 0])
            loc_cmd[0] = self.speed
            rot_cmd[1] = -angle
        elif cmd == 'right':
            # move(loc_cmd=speed, rot_cmd=[0, angle, 0])
            loc_cmd[0] = self.speed
            rot_cmd[1] = angle
        elif cmd == 'forward':
            # move(loc_cmd=speed)
            loc_cmd[0] = self.speed
        elif cmd == 'backward':
            # move(loc_cmd=-speed)
            loc_cmd[0] = -self.speed

        reward = self.move(loc_cmd=loc_cmd, rot_cmd=rot_cmd)
        return reward

    def sim_command(self, cmd):
        if cmd == 'save_view':
            self.save_view()
        elif cmd == 'change_view':
            self.change_view()
        elif cmd == 'get_position':
            self.get_pos(print_pos=True)
        return

    def save_view(self, viewmode=None):
        if viewmode is None:
            viewmode = self.request('vget /viewmode')
        res2 = self.request('vget /camera/0/' + viewmode)
        print(res2)
        return

    def change_view(self, viewmode=None):
        if viewmode is None:
            switch = dict(lit='normal', normal='depth', depth='object_mask', object_mask='lit')
            res = self.request('vget /viewmode')
            res2 = self.request('vset /viewmode ' + switch[res])
            # print(res2)
        elif viewmode in {'lit', 'normal', 'depth', 'object_mask'}:
            res2 = self.request('vset /viewmode ' + viewmode)
        return

    def get_pos(self, print_pos=False):

        if len(self.trajectory) == 0:
            rot = [float(v) for v in self.request('vget /camera/0/rotation').split(' ')]
            loc = [float(v) for v in self.request('vget /camera/0/location').split(' ')]
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
        res1 = self.request('vset /camera/0/rotation {:.3f} {:.3f} {:.3f}'.format(*new_rot))
        assert res1
        res2 = self.request('vset /camera/0/moveto {:.2f} {:.2f} {:.2f}'.format(*new_loc))
        assert res2

        return

    def request(self, message):

        res = self.client.request(message)
        # if res in 'None', try restarting sim
        while not res:
            print('[{}] sim error while trying to request {}'.format(self.name, message))
            self.reconnect()
            self.reset_agent()
            res = self.client.request(message)

        return res

    def move(self, loc_cmd=(0.0, 0.0, 0.0), rot_cmd=(0.0, 0.0, 0.0), relative=True):
        if relative:
            loc, rot = self.get_pos()
            new_rot = [sum(x) % 360 for x in zip(rot, rot_cmd)]
            displacement = [loc_cmd[0] * math.cos(math.radians(rot[1])), loc_cmd[0] * math.sin(math.radians(rot[1])), 0.0]
            new_loc = [sum(x) for x in zip(loc, displacement)]
        else:
            new_rot = rot_cmd
            new_loc = loc_cmd
        collision = False

        if rot_cmd != (0.0, 0.0, 0.0) or not relative:
            res = self.request('vset /camera/0/rotation {:.3f} {:.3f} {:.3f}'.format(*new_rot))
            assert(res == 'ok')
        if loc_cmd != (0.0, 0.0, 0.0) or not relative:
            res = self.request('vset /camera/0/moveto {:.2f} {:.2f} {:.2f}'.format(*new_loc))
            if res != 'ok':
                collision = True
                new_loc = [float(v) for v in res.split(' ')]

        self.trajectory.append(dict(location=new_loc, rotation=new_rot))

        if relative:
            reward = self.calculate_reward(displacement, collision)
        else:
            reward = 0

        return reward

    def calculate_reward(self, displacement, collision=False):
        reward = 0
        loc = np.array(self.trajectory[-1]['location'])
        prev_loc = np.array(self.trajectory[-2]['location'])
        disp = np.array(displacement)
        goal_distance = np.linalg.norm(np.subtract(loc, self.goal_location))
        if goal_distance < 200.0:  # closer than 2 meter to the goal
            return self.goal_direction_reward  # TODO: terminate episode!
        norm_displacement = np.array(displacement) / self.speed
        norm_goal_vector = np.subtract(self.goal_location, prev_loc)\
                           / np.linalg.norm(np.subtract(self.goal_location, prev_loc))
        reward += np.dot(norm_goal_vector, norm_displacement) * self.goal_direction_reward
        if collision:
            reward += self.crash_reward
            self.episode_finished = True

        return reward

    @staticmethod
    def _read_npy(res):
        return np.load(StringIO.StringIO(res))

    @staticmethod
    def _read_png(res):
        img = PIL.Image.open(StringIO.StringIO(res))
        return np.asarray(img)

    def get_observation(self, grayscale=False, show=False, viewmode='lit'):
        if viewmode == 'depth':
            res = self.request('vget /camera/0/depth npy')
            depth_image = self._read_npy(res)
            cropped = self.crop_and_resize(depth_image)
            observation = self.quantize_depth(cropped)
        else:
            res = self.request('vget /camera/0/lit png')
            rgba = self._read_png(res)
            rgb = rgba[:, :, :3]
            normalized = (rgb - 127.5) / 127.5
            if grayscale is True:
                observation = np.mean(normalized, 2)
            else:
                observation = normalized

        if show:
            # img = PIL.Image.fromarray(observation)
            # img.show()
            plt.imshow(observation)
            plt.savefig('depth.png')
            print('plot saved')

        return observation

    @staticmethod
    def quantize_depth(depth_image):    # DEBUG!!!
        bins = [0, 1, 2, 3, 4, 5, 6, 7]
        out = np.digitize(depth_image, bins) - np.ones(depth_image.shape, dtype=np.int8)
        return out

    @staticmethod
    def crop_and_resize(depth_image):
        # resize 84x84 to 16x16, crop center 8x16
        cropped = depth_image[21:63]
        resized = zoom(cropped, [0.095, 0.19], order=1)
        return resized

    def new_episode(self):
        # simple respawn: just turn around 180+/-60 deg
        # self.move(rot_cmd=(0.0, randint(120, 240), 0.0))

        # choose random respawn and goal locations
        idx_start, idx_goal = sample(range(0, len(self.locations)-1), 2)
        start_loc = (self.locations[idx_start]['x'], self.locations[idx_start]['y'], self.locations[idx_start]['z'])
        self.request('vset /camera/0/location {:.2f} {:.2f} {:.2f}'.format(*start_loc))   # teleport agent
        self.goal_location = np.array([self.locations[idx_goal]['x'], self.locations[idx_goal]['y'], self.locations[idx_goal]['z']])
        random_heading = (0.0, randint(0, 360), 0.0)
        self.request('vset /camera/0/rotation {:.3f} {:.3f} {:.3f}'.format(*random_heading))

        # reset trajectory
        self.trajectory = []
        self.get_pos()

        self.episode_finished = False
        return

    def is_episode_finished(self):
        return self.episode_finished

    def get_goal_direction(self):
        location = np.array(self.trajectory[-1]['location'])
        goal_vector = np.subtract(self.goal_location, location)
        # norm_goal_vector = goal_vector / np.linalg.norm(goal_vector)

        hdg = self.trajectory[-1]['rotation'][1]
        goal = math.degrees(math.atan2(goal_vector[1], goal_vector[0]))
        if goal < 0:
            goal += 360

        # sin(heading_error) is sufficient for directional input
        relative = math.sin(math.radians(goal - hdg))
        return np.expand_dims(np.expand_dims(relative, 0), 0)
