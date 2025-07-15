import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from act.policy import ACTPolicy, CNNMLPPolicy
from utils import set_seed # helper functions

from scipy.spatial.transform import Rotation as R

import yaml

import robosuite
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from robosuite.controllers import load_composite_controller_config

# from data_collection.scripted_policy.policy_player_stack import PolicyPlayerStack

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = True
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = True

    # Get config
    with open(args['config_file'], 'r') as f:
        config = yaml.safe_load(f)

    task_config = config['task_parameters']
    env_config = config['env_parameters']

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    state_category = task_config['state_category']
    action_category = task_config['action_category']

    # fixed parameters
    state_dim = 6
    action_dim = 7
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'env_config': env_config,
        'state_category': state_category,
        'action_category': action_category,
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len'] * 0.5
    task_name = config['task_name']
    # temporal_agg = config['temporal_agg']

    temporal_agg = True
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['state_mean']) / stats['state_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    env_config = config['env_config']
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
    
    print("env loaded")

    query_frequency = 1
    num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    robot0_base_id = env.sim.model.body_name2id("robot0_base")
    robot0_base_pos = env.sim.data.body_xpos[robot0_base_id]
    robot0_base_ori_rotm = env.sim.data.body_xmat[robot0_base_id].reshape((3,3)) # rotation matrix

    state_category = config['state_category']
    action_category = config['action_category']

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        np.random.seed(rollout_id)
        obs = reset_env(env)

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, action_dim]).cuda()

        state_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        state_list = []
        action_list = []
        success_list = []
        with torch.inference_mode():
            for t in range(max_timesteps):

                # process observations
                state_numpy, curr_image, cube_pose = process_obs(env, obs, camera_names, robot0_base_pos, robot0_base_ori_rotm, state_category)

                # # show curr image
                # plt.figure()
                # plt.imshow(curr_image[0].transpose(1, 2, 0))
                # plt.show()
                
                state = pre_process(state_numpy)
                state = torch.from_numpy(state).float().cuda().unsqueeze(0)
                state_history[:, t] = state
                curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(state, curr_image)
                    if temporal_agg:
                        # print("using temporal aggregation")
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(state, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)

                action = process_action(action, obs, robot0_base_pos, robot0_base_ori_rotm, action_category, cube_pose)
                
                ### step the environment
                obs, reward, done, info = env.step(action)
                # plt.show()

                ### for visualization
                state_list.append(state_numpy)
                action_list.append(action)

def vee_map(mat):
    """
    Convert a skew-symmetric matrix to a vector.
    :param mat: 3x3 skew-symmetric matrix
    :return: 3x1 vector
    """
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])

def get_cube_pose(env, robot0_base_pos, robot0_base_ori_rotm, cube_id):
    cube_pos_world = env.sim.data.body_xpos[cube_id]
    cube_rotm_world = env.sim.data.body_xmat[cube_id].reshape((3,3)) # rotation matrix

    cube_pos = robot0_base_ori_rotm.T @ (cube_pos_world - robot0_base_pos)
    cube_rotm = robot0_base_ori_rotm.T @ cube_rotm_world

    return cube_pos, cube_rotm

def process_obs(env, obs, camera_names, robot0_base_pos, robot0_base_ori_rotm, state_category):
    eef_pos = robot0_base_ori_rotm.T @ (obs['robot0_eef_pos'] - robot0_base_pos)
    eef_rotm = robot0_base_ori_rotm.T @ R.from_quat(obs['robot0_eef_quat_site']).as_matrix()

    eef_rotvec = R.from_matrix(eef_rotm).as_rotvec()

    name = "gripper0_right_grip_site"
    N_dof = env.robots[0].init_qpos.shape[0]
    
    jacp = env.sim.data.get_site_jacp(name)[:,:N_dof]
    jacr = env.sim.data.get_site_jacr(name)[:,:N_dof]

    J_full = np.zeros((6, N_dof))
    J_full[:3, :] = jacp
    J_full[3:, :] = jacr

    J_body = np.block([[eef_rotm.T, np.zeros((3, 3))], [np.zeros((3, 3)), eef_rotm.T]]) @ J_full

    eef_vel_body = J_body @ obs['robot0_joint_vel']

    eef_pose = np.concatenate((eef_pos, eef_rotvec))


    cubeA_main_id = env.sim.model.body_name2id("cubeA_main")
    cubeB_main_id = env.sim.model.body_name2id("cubeB_main")

    R_be_home = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, -1]])
    
    pos_cubeA, rotm_cubeA = get_cube_pose(env, robot0_base_pos, robot0_base_ori_rotm, cubeA_main_id)
    pos_cubeB, rotm_cubeB = get_cube_pose(env, robot0_base_pos, robot0_base_ori_rotm, cubeB_main_id)

    cube_pose = {"cubeA": {"pos": pos_cubeA, "rotm": rotm_cubeA},
                 "cubeB": {"pos": pos_cubeB, "rotm": rotm_cubeB}}

    rotm_list_cubeA = []
    rotm_list_cubeB = []
    for i in range(4):
        rotm_list_cubeA.append(rotm_cubeA @ R.from_euler("z", i * np.pi / 2).as_matrix())
        rotm_list_cubeB.append(rotm_cubeB @ R.from_euler("z", i * np.pi / 2).as_matrix())
    
    # find the rotation matrix that has a minimum distance to the R_be_home
    min_dist_cubeA = 100 # any large number bigger than 4
    min_dist_cubeB = 100

    for i in range(4):
        dist_cubeA = np.trace(np.eye(3) - R_be_home.T @ rotm_list_cubeA[i])
        dist_cubeB = np.trace(np.eye(3) - R_be_home.T @ rotm_list_cubeB[i])

        if dist_cubeA < min_dist_cubeA:
            min_dist_cubeA = dist_cubeA
            rotm_cubeA = rotm_list_cubeA[i]

        if dist_cubeB < min_dist_cubeB:
            min_dist_cubeB = dist_cubeB
            rotm_cubeB = rotm_list_cubeB[i]

    imgs = []
    for cam_name in camera_names:
        img = rearrange(obs[f"{cam_name}_image"], 'h w c -> c h w')
        imgs.append(img)
    
    img_stack = np.stack(imgs, axis=0)

    if state_category == "eef_pose_world":

        state = eef_pose

    elif state_category == "GCEV":
        pd = pos_cubeA
        Rd = rotm_cubeA

        ep = eef_rotm.T @ (eef_pos - pd)
        eR = vee_map(Rd.T @ eef_rotm - eef_rotm.T @ Rd)

        state = np.concatenate((ep, eR))


    elif state_category == "eef_vel":
        state = eef_vel_body

    # print("state:", state)

    return state, img_stack, cube_pose

def process_action(action, obs, robot0_base_pos, robot0_base_ori_rotm, action_category, cube_pose):
    eef_pos = robot0_base_ori_rotm.T @ (obs['robot0_eef_pos'] - robot0_base_pos)
    eef_rotm = robot0_base_ori_rotm.T @ R.from_quat(obs['robot0_eef_quat_site']).as_matrix()

    if action_category == "eef_pose_world":
        action_return = action

    elif action_category == "relative":
        g = np.eye(4)
        g[:3, 3] = eef_pos
        g[:3, :3] = eef_rotm

        g_rel = np.eye(4)
        g_rel[:3, 3] = action[0:3]
        g_rel[:3, :3] = R.from_rotvec(action[3:6]).as_matrix()

        g_a = g @ g_rel

        action_return = np.zeros(7)
        action_return[0:3] = g_a[:3, 3]
        action_return[3:6] = R.from_matrix(g_a[:3, :3]).as_rotvec()
        # gripper action
        action_return[-1] = action[-1]

    elif action_category == "desired_frame":
        cubeA_pos = cube_pose["cubeA"]["pos"]
        cubeA_rotm = cube_pose["cubeA"]["rotm"]

        gd = np.eye(4)
        gd[:3, 3] = cubeA_pos
        gd[:3, :3] = cubeA_rotm

        g_rel = np.eye(4)
        g_rel[:3, 3] = action[0:3]
        g_rel[:3, :3] = R.from_rotvec(action[3:6]).as_matrix()

        g_a = gd @ g_rel

        action_return = np.zeros(7)
        action_return[0:3] = g_a[:3, 3]
        action_return[3:6] = R.from_matrix(g_a[:3, :3]).as_rotvec()
        # gripper action
        action_return[-1] = action[-1]

    return action_return


def reset_env(env):
    obs = env.reset()

    """
    possible body names:
    'world', 'table', 'left_eef_target', 'right_eef_target', 
    'robot0_base', 'robot0_link0', 'robot0_link1', 'robot0_link2', 'robot0_link3', 'robot0_link4', 'robot0_link5', 'robot0_link6', 'robot0_link7', 
    'robot0_right_hand', 'gripper0_right_right_gripper', 'gripper0_right_eef', 'gripper0_right_leftfinger', 'gripper0_right_finger_joint1_tip', 
    'gripper0_right_rightfinger', 'gripper0_right_finger_joint2_tip', 'fixed_mount0_base', 'fixed_mount0_controller_box', 
    'fixed_mount0_pedestal_feet', 'fixed_mount0_torso', 'fixed_mount0_pedestal', 'cubeA_main', 'cubeB_main'
    """

    R_be_home = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, -1]])

    robot0_base_id = env.sim.model.body_name2id("robot0_base")
    robot0_base_pos = env.sim.data.body_xpos[robot0_base_id]
    robot0_base_ori_rotm = env.sim.data.body_xmat[robot0_base_id].reshape((3,3)) # rotation matrix

    # fix Cubes' position and orientation for the testing
    cubeA_main_id = env.sim.model.body_name2id("cubeA_main")
    cubeB_main_id = env.sim.model.body_name2id("cubeB_main")

    pos_cubeA_world = np.array([0.018, 0.068, 0.8225])
    rotm_cubeA_world = np.array([[ 0.90203619, -0.43166042,  0.        ],
                                [ 0.43166042,  0.90203619,  0.        ],
                                [ 0.        ,  0.        ,  1.        ]])

    pos_cubeB_world = np.array([-0.06605931, -0.07676506,  0.83      ])
    rotm_cubeB_world = np.array([[ 0.49611262,  0.86825818 , 0.        ],
                                [-0.86825818,  0.49611262 , 0.        ],
                                [ 0.        ,  0.         , 1.        ]])

    # Set the position and orientation of the cubes in the world frame
    env.sim.model.body_pos[cubeA_main_id] = pos_cubeA_world
    env.sim.model.body_quat[cubeA_main_id] = R.from_matrix(rotm_cubeA_world).as_quat(scalar_first=True)
    env.sim.model.body_pos[cubeB_main_id] = pos_cubeB_world
    env.sim.model.body_quat[cubeB_main_id] = R.from_matrix(rotm_cubeB_world).as_quat(scalar_first = True)

    ###################################################

    cubeA_main_id = env.sim.model.body_name2id("cubeA_main")
    pos_cubeA_world = env.sim.data.body_xpos[cubeA_main_id]
    rotm_cubeA_world = env.sim.data.body_xmat[cubeA_main_id].reshape((3,3)) # rotation matrix

    pos_cubeA = robot0_base_ori_rotm.T @ (pos_cubeA_world - robot0_base_pos)
    rotm_cubeA = robot0_base_ori_rotm.T @ rotm_cubeA_world

    cubeB_main_id = env.sim.model.body_name2id("cubeB_main")
    pos_cubeB_world = env.sim.data.body_xpos[cubeB_main_id]
    rotm_cubeB_world = env.sim.data.body_xmat[cubeB_main_id].reshape((3,3)) # rotation matrix

    pos_cubeB = robot0_base_ori_rotm.T @ (pos_cubeB_world - robot0_base_pos)
    rotm_cubeB = robot0_base_ori_rotm.T @ rotm_cubeB_world


    rand_xy_A = np.random.uniform(-0.003, 0.003, size=(2,))
    rand_xy_B = np.random.uniform(-0.01, 0.01, size=(2,))
    rand_z_A = np.random.uniform(-0.005, 0)

    pos_cubeA = pos_cubeA + rotm_cubeA @ np.array([0, 0, -0.00625]) + np.array([rand_xy_A[0], rand_xy_A[1], rand_z_A - 0.005])
    pos_cubeB = pos_cubeB + rotm_cubeB @ np.array([0, 0, -0.01]) + np.array([rand_xy_B[0], rand_xy_B[1], 0])    

    rand_init_pos = np.random.uniform(-0.066, 0.06, size=(3,))
    init_pos = pos_cubeA + np.array([rand_init_pos[0], rand_init_pos[1], 0.1])

    rand_init_euler = np.random.uniform(-10/180*np.pi, 10/180*np.pi, size=(3,))
    init_rotm = rotm_cubeA @ R.from_euler("xyz", rand_init_euler).as_matrix()

#Rotation matrix post processing for cube A and cube B
    rotm_x = R.from_euler("x", np.pi).as_matrix()
    rotm_cubeA = rotm_cubeA @ rotm_x
    rotm_cubeB = rotm_cubeB @ rotm_x 

    # rotate rotm_cubeA and rotm_cubeB with 90 degrees interval, choose the rotation matrix has a minmum distance to the R_be_home
    rotm_list_cubeA = []
    rotm_list_cubeB = []
    for i in range(4):
        rotm_list_cubeA.append(rotm_cubeA @ R.from_euler("z", i * np.pi / 2).as_matrix())
        rotm_list_cubeB.append(rotm_cubeB @ R.from_euler("z", i * np.pi / 2).as_matrix())
    
    # find the rotation matrix that has a minimum distance to the R_be_home
    min_dist_cubeA = 100 # any large number bigger than 4
    min_dist_cubeB = 100

    for i in range(4):
        dist_cubeA = np.trace(np.eye(3) - R_be_home.T @ rotm_list_cubeA[i])
        dist_cubeB = np.trace(np.eye(3) - R_be_home.T @ rotm_list_cubeB[i])

        if dist_cubeA < min_dist_cubeA:
            min_dist_cubeA = dist_cubeA
            rotm_cubeA = rotm_list_cubeA[i]

        if dist_cubeB < min_dist_cubeB:
            min_dist_cubeB = dist_cubeB
            rotm_cubeB = rotm_list_cubeB[i]

    rand_init_pos = np.random.uniform(-0.08, 0.08, size=(3,))
    init_pos = pos_cubeA + np.array([rand_init_pos[0], rand_init_pos[1], 0.1])

    rand_init_euler = np.random.uniform(-10/180*np.pi, 10/180*np.pi, size=(3,))
    init_rotm = rotm_cubeA @ R.from_euler("xyz", rand_init_euler).as_matrix()

    n_action = 7
    arrived = False
    for i in range(100):
        if arrived:
            break
        action = np.zeros(int(n_action))
        action[0:3] = init_pos
        action[3:6] = R.from_matrix(init_rotm).as_rotvec()
        action[6] = -1
        # arrived = self.check_arrived(self.pos_cubeA, self.rotm_cubeA, init_pos, init_rotm, 0.001)

        obs, reward, done, info = env.step(action)

    return obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    parser.add_argument('--config_file', type=str, default="config/train/ACT_stack.yaml")
    
    main(vars(parser.parse_args()))
