from config import Config
from time import sleep
import numpy as np
import yaml
import os


def coordinate_evaluation(worker_list, coord):

    # START EVAL
    for w in worker_list:
        w.ongoing_evaluation = True
        w.start_new_eval_episode = True
    worker_list[0].start_eval = False

    eval_finished = False
    started_episodes = 0
    while not eval_finished:
        try:
            sleep(0.1)
            started_episode_count = 0
            for w in worker_list:
                started_episode_count += w.started_eval_episodes
            if started_episode_count > started_episodes:
                started_episodes = started_episode_count
                print('Started {} episodes.'.format(started_episodes))
            if started_episode_count > Config.MAX_EPISODES_FOR_EVAL - 1:
                print('All {} episodes started.'.format(started_episodes))
                eval_finished = True
        except KeyboardInterrupt:
            print('terminating threads.....')
            coord.request_stop()

    # WAIT - we started enough episodes, now just need to wait for finishing them
    for w in worker_list:
        w.start_new_eval_episode = False

    all_episodes_ended = False
    finished_episodes = 0
    while not all_episodes_ended:
        try:
            sleep(0.1)
            finished_episode_count = 0
            for w in worker_list:
                finished_episode_count += w.finished_eval_episodes
            if finished_episode_count > finished_episodes:
                finished_episodes = finished_episode_count
                print('Waiting for {} episodes to complete.'.format(
                    Config.MAX_EPISODES_FOR_EVAL - finished_episode_count))
            if finished_episode_count >= Config.MAX_EPISODES_FOR_EVAL:
                all_episodes_ended = True
        except KeyboardInterrupt:
            print('terminating threads.....')
            coord.request_stop()

    # FINISH - shut evaluation down, collect data and save
    cumulative_rewards = np.array([])
    depth_losses = np.array([])
    rewards = []
    episode_lengths = np.array([])
    trajectories = []
    for w in worker_list:
        w.ongoing_evaluation = False
        w.started_eval_episodes = 0
        print('[{}] did {} episodes'.format(w.name, w.finished_eval_episodes))
        w.finished_eval_episodes = 0
        # collect eval data from workers and save it
        cumulative_rewards = np.append(cumulative_rewards, w.eval_cumulative_rewards)
        w.eval_cumulative_rewards = np.array([])
        depth_losses = np.append(depth_losses, w.eval_depth_losses)
        w.eval_depth_losses = np.array([])
        for rws in w.eval_rewards:
            rewards.append(rws)
        w.eval_rewards = []
        for traj in w.eval_trajectories:
            trajectories.append(traj)
            episode_lengths = np.append(episode_lengths, np.array(len(traj['traj'])))
        w.eval_trajectories = []
    eval_step = worker_list[0].last_model_save_steps

    eval_path = './EVAL/{}k'.format(int(eval_step / 1))
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    # save csv
    with open('./EVAL/learning_curve.csv', 'a+') as lc_file:
        lc_file.write('{}, {}, {}, {}\n'.format(eval_step,
                                                cumulative_rewards.mean(),
                                                depth_losses.mean(),
                                                episode_lengths.mean()))

    with open('{}/lc_traj_{}.yaml'.format(eval_path, int(eval_step/1)), 'a+') as traj_yaml:
        yaml.dump(trajectories, stream=traj_yaml, default_flow_style=False)

    with open('./EVAL/learning_curve_data.yaml', 'a+') as data_yaml:
        yaml.dump([{'step': int(eval_step),
                    'cumulative_rewards': cumulative_rewards.tolist(),
                    'depth_losses': depth_losses.tolist(),
                    'episode_lengths': episode_lengths.tolist()}],
                  stream=data_yaml, default_flow_style=False)

    print('--- Integrated evaluation finished.')
