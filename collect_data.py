import numpy as np
import argparse
import yaml
import pickle
import timeit
import os

import robosuite

import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from robosuite.controllers import load_composite_controller_config

from scripted_policy.policy_player_stack import PolicyPlayerStack

def main(args):
    # Get config
    with open(args['config_file'], 'r') as f:
        config = yaml.safe_load(f)
    env_config = config['env_parameters']
    task_config = config['task_parameters']

    # setup the environment
    controller_config = load_composite_controller_config(robot=env_config['robots'][0], controller=env_config['controller'])
    env = robosuite.make(
    env_config['env_name'],
    robots=env_config['robots'][0],
    controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    has_renderer=False,                      # on-screen rendering
    render_camera=None,              # visualize the "frontview" camera
    has_offscreen_renderer=True,           # no off-screen rendering                       
    horizon=env_config['max_iter'],                            # each episode terminates after 200 steps
    use_object_obs=False,                   # no observations needed
    use_camera_obs=True,
    camera_names=env_config['camera_names'],
    camera_heights=env_config['camera_heights'],
    camera_widths = env_config['camera_widths'],
    camera_depths = env_config['camera_depths'],
    control_freq=env_config['control_freq'],                       # 20 hz control for applied actions
    fix_initial_cube_pose = env_config['fix_initial_cube_pose'],
    training = True,
    )

    save = True

    # setup the scripted policy
    if env_config['env_name'] == "Stack" or env_config['env_name'] == "StackCustom":
        player = PolicyPlayerStack(env, render = False, randomized = True)

    else:
        pass

    dataset_dir = task_config['raw_dataset_dir']
    num_episodes = task_config['num_episodes']
    max_episodes = 1000

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    current_episode = 0

    for i in range(max_episodes):
        if current_episode >= num_episodes:
            break
        rollout, done = player.get_demo(seed = i)
        print(f"Episode: {current_episode}, Success: {done}, length: {len(rollout['observations'])}")
        # save the rollout using pickle
        if save and done:
            current_episode += 1
            with open(os.path.join(dataset_dir, f"demo_{current_episode}.pkl"), 'wb') as f:
                pickle.dump(rollout, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ACT")
    parser.add_argument("--controller", type=str, default="config/controller/indy7_absolute_pose.json")
    parser.add_argument("--config_file", type=str, default="config/train/ACT_stack_08.yaml")

    args = parser.parse_args()

    main(vars(args))