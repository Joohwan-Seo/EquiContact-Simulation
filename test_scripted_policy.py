import numpy as np
import argparse
import yaml
import pickle
import timeit
import os

import robosuite

import robosuite.macros as macros
# macros.IMAGE_CONVENTION = "opencv"
from robosuite.controllers import load_composite_controller_config

from scripted_policy.policy_player_stack import PolicyPlayerStack
from scripted_policy.policy_player_pih import PolicyPlayerPIH

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
    has_renderer=True,                      # on-screen rendering
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
    )

    # setup the scripted policy
    if env_config['env_name'] == "Stack" or env_config['env_name'] == "StackCustom":
        player = PolicyPlayerStack(env, render = False, randomized = True, debug = False, save = False)

    elif env_config['env_name'] == "PegInHole":
        player = PolicyPlayerPIH(env, render = False, randomized = False, debug = True, save = False)
    else:
        raise NotImplementedError("Selected environment is not implemented for scripted policy")

    player.get_demo(seed = 5)

    quit()
    # for i in range(1):
        # rollout = player.get_demo(seed = i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ACT")
    parser.add_argument("--controller", type=str, default="config/controller/indy7_absolute_pose.json")
    parser.add_argument("--config_file", type=str, default="config/train/ACT_stack.yaml")

    args = parser.parse_args()

    main(vars(args))