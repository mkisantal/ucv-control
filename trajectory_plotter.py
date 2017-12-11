import matplotlib.pyplot as plt
import yaml
import numpy as np
from config import Configuration
import os

# COORDINATES ARE INVERTED!!!
# as UE4 uses left-handed coordinate system


# TODO: load reward instead of recalculating it here
def calculate_reward(traj, idx, goal):
    loc = np.array(traj[idx]['location'][:2])
    prev_loc = np.array(traj[idx-1]['location'][:2])
    disp = np.subtract(loc, prev_loc)
    if np.linalg.norm(disp) == 0.0:
        return 0
    goal_distance = np.linalg.norm(np.subtract(loc, goal))
    if goal_distance < 200.0:  # closer than 2 meter to the goal
        return 1  # TODO: terminate episode!
    norm_displacement = np.array(disp) / np.linalg.norm(disp)
    norm_goal_vector = np.subtract(goal, prev_loc) \
                       / np.linalg.norm(np.subtract(goal, prev_loc))
    reward = np.dot(norm_goal_vector, norm_displacement) * 1

    return reward


def draw_labyrinth(axis):
    # quick and dirty code to draw the labyrinth environment

    # draw the map
    y = [-220, 1460, 1460, -220, -220]
    x = [-440, -440, 1240, 1240, -440]
    line, = axis.plot(x, y, 'k-')

    doors = [[True, True, True, True],
             [False, True, False, True],
             [True, True, False, True]]

    # walls parallel with y direction
    for i in range(3):
        x12 = 210 + i * 420
        y1 = -440
        y2 = 1240

        y = [x12, x12]
        x = [y1, y2]
        line, = axis.plot(x, y, 'k-')

        # overdrawing doors
        for j in range(4):
            if doors[i][j]:
                x = [-280 + j * 420, -280 + j * 420 + 120]
                line, = axis.plot(x, y, 'w-')
                y2 = [x12 - 5, x12 + 5]
                # door side
                line, = axis.plot([x[0], x[0]], y2, 'k-')
                line, = axis.plot([x[1], x[1]], y2, 'k-')

    doors = [[False, True, True, True],
             [True, True, False, True],
             [True, False, True, True]]

    # walls parallel with x direction
    for i in range(3):
        y12 = -10 + i * 420
        x1 = -220
        x2 = 1460

        y = [x1, x2]
        x = [y12, y12]
        line, = axis.plot(x, y, 'k-')
        for j in range(4):
            if doors[i][j]:
                y = [-60 + j * 420, -60 + j * 420 + 120]
                line, = axis.plot(x, y, 'w-')
                x2 = [y12 - 5, y12 + 5]
                line, = axis.plot(x2, [y[0], y[0]], 'k-')
                line, = axis.plot(x2, [y[1], y[1]], 'k-')

    obstacles = [[630, -130],
                 [200, 460],
                 [1050, 850],
                 [180, 860],
                 [630, 1130],
                 [1030, 1250]]
    for obstacle in obstacles:
        axis.plot(obstacle[0], obstacle[1], 'ks')


def draw_forest(axis):
        with open('./map2.yaml', 'r') as map_file:
            obstacles = yaml.load(map_file)
        for obstacle in obstacles:
            axis.plot(obstacle[0], obstacle[1], 'ks', markersize=0.2)


def interpolate_color(value):
    if value < 0:
        value = 0
    if value > 1:
        value = 1
    r, g, b = 0.5 - 0.5*value, 0, 0.5 + 0.5*value
    return r, g, b


def draw_trajectory(traj, axis, config, show_start=False, show_crash=False, goal=None):

    if config.VERBOSITY > 1:
        print('Drawing trajectory with {} points.'.format(len(traj)))
    x = []
    y = []

    if show_start:
        axis.plot(traj[0]['location'][1], traj[0]['location'][0], 'go')

    for i in range(len(traj)):
        x.append(traj[i]['location'][1])
        y.append(traj[i]['location'][0])
        if goal is not None and i is not 0:
            reward = calculate_reward(traj, i, goal)
            line, = axis.plot(x[-2:], y[-2:], '-', alpha=0.7, color=interpolate_color(reward), linewidth=0.5)  # 'go-'

    if show_crash:
        axis.plot(traj[-1]['location'][1], traj[-1]['location'][0], 'ro', markersize=1)

    if goal is not None:
        axis.plot(goal[1], goal[0], 'yo')


def main(global_steps):
    config = Configuration('eval', 0)
    if config.VERBOSITY > 0:
        print('Evaluation: plotting trajectories...')

    # loading trajectory files
    trajectories = []
    for i in range(config.NUM_WORKERS):
        filename = './TRAJECTORIES/' + global_steps + '/trajectory_player_{}.yaml'.format(i)
        if config.VERBOSITY > 1:
            print('loading trajectory_player_{}.yaml'.format(i))
        with open(filename, 'r') as trajectory_file:
            data = yaml.load(trajectory_file)
        trajectories.extend(data)

    fig = plt.figure()
    fig.suptitle(global_steps + ' global steps')
    ax = fig.add_subplot(111, aspect='equal')
    plt.ion()
    # draw_labyrinth(ax)
    draw_forest(ax)

    counter = 0
    for trajectory in trajectories:
        counter += 1
        if counter > 50:
            pass
        else:
            draw_trajectory(trajectory['traj'], ax, config, show_start=True, show_crash=True, goal=trajectory['goal'])

    if not os.path.exists('./EVAL'):
        os.makedirs('./EVAL')

    fig_name = './EVAL/{}.pdf'.format(global_steps)
    fig.savefig(fig_name, format='pdf')
    if config.VERBOSITY > 0:
        print('Evaluation: {} trajectories plotted at {} global steps, saved to {}'.format(counter, global_steps, fig_name))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--global_steps',
                        help='Select which set of trajectories to use (i.e. trajectories done at \'50k\' global steps)',
                        default='50k')
    args = parser.parse_args()

    main(global_steps=args.global_steps)
