# action
import math
import numpy as np
import StringIO
from PIL import Image
from random import randint, sample
import time
import subprocess
import ucv_utils
import os
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import yaml
from unrealcv import Client


class Commander:

    """ Class for interacting with an UE4 Game packaged with the UnrealCV plugin"""

    def __init__(self, number, config, name, mode=None):
        self.config = config
        self.trajectory = []
        self.name = name
        self.port = 7000 + number * 100
        self.number = number

        # navigation goal direction
        self.goal_location = None
        self.goal_vector = None

        # list of start and goal locations
        self.locations = []
        if self.config.RANDOM_SPAWN_LOCATIONS:
            self.locations = None
        else:
            with open(self.config.SIM_DIR_LIST[number] + 'locations.yaml', 'r') as loc_file:
                self.locations = yaml.load(loc_file)

        # RL rewards
        self.goal_direction_reward = 1.0
        self.crash_reward = -10.0
        self.control_effort_reward_multiplier = -1.0/100

        # Agent actions
        self.action_space = ('ang_acc_left', 'ang_acc_right', 'ang_acc_forward')  # each has to be defined in action()
        self.state_space_size = self.config.STATE_SHAPE  # for now RGB
        self.speed = 40.0  # cm/step    # TODO: move these to config
        self.angular_speed_state = 0
        self.angular_acc = 5  # deg/step
        self.max_angular_speed = 15  # deg/step

        self.episode_finished = False
        self.should_terminate = False

        # load UE4 game
        self.sim = None
        self.client = None
        self.should_stop = False
        self.mode = mode
        if self.mode == 'test':
            self.client = Client((self.config.HOST, self.config.PORT + number))
            self.client.connect()
        else:
            while not self.start_sim():
                pass

    def shut_down(self):

        """ Disconnect client and terminate the game. """

        if self.client.isconnected():
            self.client.disconnect()
        if self.sim is not None:
            self.sim.terminate()
            self.sim = None

    def start_sim(self, restart=False):

        """ Starting game and connecting the UnrealCV client """

        if self.sim is not None:
            # disconnect and terminate if restarting
            # self.shut_down()
            self.sim.terminate()
        self.port += 1
        ucv_utils.set_port(self.port, self.config.SIM_DIR_LIST[self.number], self.config)
        time.sleep(2)
        print('[{}] Connection attempt on PORT {}.'.format(self.name, self.port))
        with open(os.devnull, 'w') as fp:   # Sim messages on stdout are discarded
            self.sim = subprocess.Popen(self.config.SIM_DIR_LIST[self.number] + self.config.SIM_NAME, stdout=fp)
        time.sleep(5)
        self.client = Client((self.config.HOST, self.port))
        time.sleep(2)
        self.client.connect()
        time.sleep(2)
        got_connection = self.client.isconnected()
        if got_connection:
            if restart:
                self.reset_agent()
            else:
                self.new_episode()
            return True
        else:
            return False

    def action(self, cmd):

        """ Mapping the actions of the RL agent to displacement and rotation commands. """

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
        elif cmd == 'ang_acc_left':
            loc_cmd[0] = self.speed
            self.angular_speed_state -= self.angular_acc
            if abs(self.angular_speed_state) > self.max_angular_speed:
                self.angular_speed_state = -self.max_angular_speed
            rot_cmd[1] = self.angular_speed_state
        elif cmd == 'ang_acc_right':
            loc_cmd[0] = self.speed
            self.angular_speed_state += self.angular_acc
            if abs(self.angular_speed_state) > self.max_angular_speed:
                self.angular_speed_state = self.max_angular_speed
            rot_cmd[1] = self.angular_speed_state
        elif cmd == 'ang_acc_forward':
            loc_cmd[0] = self.speed
            rot_cmd[1] = self.angular_speed_state
        else:
            raise ValueError('Unknown action [{}]'.format(cmd))

        control_effort_reward = abs(self.angular_speed_state) * self.control_effort_reward_multiplier
        movement_reward = self.move(loc_cmd=loc_cmd, rot_cmd=rot_cmd)  # acting
        total_reward = movement_reward + control_effort_reward
        return total_reward

    # def sim_command(self, cmd):
    #     if cmd == 'save_view':
    #         self.save_view()
    #     elif cmd == 'change_view':
    #         self.change_view()
    #     elif cmd == 'get_position':
    #         self.get_pos(print_pos=True)
    #     return

    # def save_view(self, viewmode=None):
    #     if viewmode is None:
    #         viewmode = self.request('vget /viewmode')
    #     res2 = self.request('vget /camera/0/' + viewmode)
    #     print(res2)
    #     return

    # def change_view(self, viewmode=None):
    #     if viewmode is None:
    #         switch = dict(lit='normal', normal='depth', depth='object_mask', object_mask='lit')
    #         res = self.request('vget /viewmode')
    #         res2 = self.request('vset /viewmode ' + switch[res])
    #         # print(res2)
    #     elif viewmode in {'lit', 'normal', 'depth', 'object_mask'}:
    #         res2 = self.request('vset /viewmode ' + viewmode)
    #     return

    def get_pos(self, print_pos=False):

        """ Get the last position from the stored trajectory, if trajectory is empty then request it from the sim. """

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

        """ Reset the agent to continue interaction in the state where it was interrupted. """

        new_loc = self.trajectory[-1]["location"]
        new_rot = self.trajectory[-1]["rotation"]
        res1 = self.request('vset /camera/0/rotation {:.3f} {:.3f} {:.3f}'.format(*new_rot))
        assert res1
        res2 = self.request('vset /camera/0/location {:.2f} {:.2f} {:.2f}'.format(*new_loc))
        assert res2

        return

    def request(self, message):

        """ Send request with UnrealCV client, if unsuccessful restart sim. """

        res = self.client.request(message)
        # if res in 'None', try restarting sim
        while not res:
            print('[{}] sim error while trying to request {}'.format(self.name, message))
            success = self.start_sim(restart=True)
            if success:
                res = self.client.request(message)

        return res

    def move(self, loc_cmd=(0.0, 0.0, 0.0), rot_cmd=(0.0, 0.0, 0.0), relative=True):

        """ Move/rotate agent, update trajectory, return immediate reward. """

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
            # assert (res == 'ok')
            final_loc = [float(v) for v in self.request('vget /camera/0/location').split(' ')]
            if final_loc != [round(v, 2) for v in new_loc]:
                collision = True
                new_loc = final_loc

        self.trajectory.append(dict(location=new_loc, rotation=new_rot))

        if relative:
            movement_reward = self.displacement_reward(displacement, collision)
        else:
            movement_reward = 0

        if collision:
            movement_reward += self.crash_reward
            self.episode_finished = True

        return movement_reward

    def displacement_reward(self, displacement):

        """ Calculate reward based on displacement and collision. """

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

        return reward

    @staticmethod
    def _read_npy(res):
        return np.load(StringIO.StringIO(res))

    @staticmethod
    def _read_png(res):
        img = Image.open(StringIO.StringIO(res))
        return np.asarray(img)

    def get_observation(self, grayscale=False, show=False, viewmode='lit'):

        """ Get image from the simulator. """

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
            # img = Image.fromarray(observation)
            # img.show()
            plt.imshow(observation)
            plt.savefig('depth.png')
            print('plot saved')

        return observation

    @staticmethod
    def quantize_depth(depth_image):

        """ Depth classes """

        bins = [0, 1, 2, 3, 4, 5, 6, 7]  # TODO: better depth bins
        out = np.digitize(depth_image, bins) - np.ones(depth_image.shape, dtype=np.int8)
        return out

    @staticmethod
    def crop_and_resize(depth_image):
        # resize 84x84 to 16x16, crop center 8x16
        cropped = depth_image[21:63]
        resized = zoom(cropped, [0.095, 0.19], order=1)
        return resized

    def random_start_location(self, heading):
        collision_at_start = True
        while collision_at_start:
            # spawn location is ok, if we can move forward a bit without colliding
            start_x = randint(self.config.MAP_X_MIN, self.config.MAP_X_MAX)
            start_y = randint(self.config.MAP_Y_MIN, self.config.MAP_Y_MAX)
            self.request('vset /camera/0/pose {} {} {} {} {} {}'.format(start_x, start_y, 150, 0, heading, 0))
            step = 50
            small_step_forward = (start_x + step * math.cos(math.radians(heading)),
                                  start_y + step * math.sin(math.radians(heading)), 150.0)
            self.request('vset /camera/0/moveto {} {} {}'.format(*small_step_forward))
            final_loc = [round(float(v), 2) for v in self.request('vget /camera/0/location').split(' ')]
            if final_loc == [round(v, 2) for v in small_step_forward]:
                # acceptable start location found
                collision_at_start = False

        return start_x, start_y

    def new_episode(self, save_trajectory=False, start=None, goal=None):

        """ Choose new start and goal locations, replace agent. """

        if save_trajectory:
            self.save_trajectory()

        # choose random respawn and goal locations, either randomly or from a list of predetermined locations
        random_heading = (0.0, randint(0, 360), 0.0)
        if self.config.RANDOM_SPAWN_LOCATIONS:
            if self.config.TRAIN_MODE:
                goal_x = randint(self.config.MAP_X_MIN, self.config.MAP_X_MAX)
                goal_y = randint(self.config.MAP_Y_MIN, self.config.MAP_Y_MAX)
                start_x, start_y = self.random_start_location(random_heading[1])
            else:
                goal_x = self.config.EVAL_GOAL_X
                goal_y = self.config.EVAL_GOAL_Y
                start_x = self.config.EVAL_START_X
                start_y = self.config.EVAL_START_Y
            start_loc = (start_x, start_y, 150)
            self.goal_location = (goal_x, goal_y, 150)
        else:
            if start is None or goal is None:
                idx_start, idx_goal = sample(range(0, len(self.locations) - 1), 2)
            else:
                idx_start = start
                idx_goal = goal
            start_loc = (self.locations[idx_start]['x'], self.locations[idx_start]['y'], self.locations[idx_start]['z'])
            self.goal_location = np.array([self.locations[idx_goal]['x'], self.locations[idx_goal]['y'], self.locations[idx_goal]['z']])

        # reset trajectory
        self.trajectory = []
        loc = [float(v) for v in start_loc]
        rot = [float(v) for v in random_heading]
        self.trajectory.append(dict(location=loc, rotation=rot))

        # teleport agent
        self.request('vset /camera/0/location {:.2f} {:.2f} {:.2f}'.format(*start_loc))   # teleport agent
        self.request('vset /camera/0/rotation {:.3f} {:.3f} {:.3f}'.format(*random_heading))

        self.episode_finished = False
        return

    def is_episode_finished(self):
        return self.episode_finished

    def get_goal_direction(self):

        """ Producing goal direction input for the agent. """

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

    def save_trajectory(self):

        """ Save trajectory for evaluation. """

        filename = './trajectory_{}.yaml'.format(self.name)
        traj_dict = {'traj': self.trajectory,
                     'goal': [float(self.goal_location[0]), float(self.goal_location[1])],
                     }  # TODO: add rewards
        with open(filename, 'a') as trajectory_file:
            yaml.dump([traj_dict], stream=trajectory_file, default_flow_style=False)


