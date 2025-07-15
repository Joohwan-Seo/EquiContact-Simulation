import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from policy import ACTPolicy, CNNMLPPolicy
from utils import set_seed # helper functions

from scipy.spatial.transform import Rotation as R

import yaml

import robosuite
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from robosuite.controllers import load_composite_controller_config
from transform_utils import hat_map, vee_map
from scipy.linalg import expm
# from data_collection.scripted_policy.policy_player_stack import PolicyPlayerStack

import IPython
e = IPython.embed

class EvalEnv():
    def __init__(self, args):
        set_seed(1)
        # command line parameters
        is_eval = True
        policy_class = args['policy_class']
        onscreen_render = args['onscreen_render']
        task_name = args['task_name']
        batch_size_train = args['batch_size']
        batch_size_val = args['batch_size']
        num_epochs = args['num_epochs']

        is_sim = True

        # Get config
        with open(args['config_file'], 'r') as f:
            config = yaml.safe_load(f)

        self.task_config = config['task_parameters']
        self.env_config = config['env_parameters']
        self.eval_config = config['eval_parameters']

        episode_len = self.eval_config['episode_len']
        camera_names = self.eval_config['camera_names']
        state_category = self.eval_config['state_category']
        action_category = self.eval_config['action_category']
        noisy_reference = self.eval_config['noisy_reference']
        ckpt_dir = self.eval_config['ckpt_dir']

        # fixed parameters
        state_dim = self.eval_config['state_dim']
        action_dim = self.eval_config['action_dim']
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
                            'state_dim': state_dim,
                            'action_dim': action_dim,
                            }
        elif policy_class == 'CNNMLP':
            policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                            'camera_names': camera_names,}
        else:
            raise NotImplementedError

        self.config = {
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
            'env_config': self.env_config,
            'state_category': state_category,
            'action_category': action_category,
            'noisy_reference': noisy_reference,
        }

        controller_config = load_composite_controller_config(robot=self.env_config['robots'][0], controller=self.env_config['controller'])

        self.env = robosuite.make(
        self.env_config['env_name'],
        robots=self.env_config['robots'][0],
        controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
        has_renderer=True,                      # on-screen rendering
        render_camera=None,              # visualize the "frontview" camera
        has_offscreen_renderer=True,           # no off-screen rendering                       
        horizon=self.env_config['max_iter'],                            # each episode terminates after 200 steps
        use_object_obs=False,                   # no observations needed
        use_camera_obs=True,
        camera_names=self.env_config['camera_names'],
        camera_heights=self.env_config['camera_heights'],
        camera_widths = self.env_config['camera_widths'],
        camera_depths = self.env_config['camera_depths'],
        control_freq=self.env_config['control_freq'],                       # 20 hz control for applied actions
        fix_initial_cube_pose = self.env_config['fix_initial_cube_pose'],
        )

        self.dt = 1.0 / self.env.control_freq


        ckpt_name = 'policy_best.ckpt'

        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.policy = self.make_policy(policy_class, policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - stats['state_mean']) / stats['state_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        print(f"action_mean: {stats['action_mean'].shape}, action_std: {stats['action_std'].shape}")

        # if self.action_category == "relative":
        #     stats_action_std_cuda = stats['action_std'].cuda()
        #     stats_action_mean_cuda = stats['action_mean'].cuda()
        #     self.post_process_cuda = lambda a: a * stats_action_std_cuda + stats_action_mean_cuda

    def make_policy(self, policy_class, policy_config):
        if policy_class == 'ACT':
            policy = ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(policy_config)
        else:
            raise NotImplementedError
        return policy
    
    def get_cube_pose(self, cube_id):
        cube_pos_world = self.env.sim.data.body_xpos[cube_id]
        cube_rotm_world = self.env.sim.data.body_xmat[cube_id].reshape((3,3)) # rotation matrix

        cube_pos = self.robot0_base_ori_rotm.T @ (cube_pos_world - self.robot0_base_pos)
        cube_rotm = self.robot0_base_ori_rotm.T @ cube_rotm_world

        return cube_pos, cube_rotm
    
    def vee_map(self, mat):
        """
        Convert a matrix to a vector using the vee operator
        """
        return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
    
    def reset_env(self):
        self.reset_flag = True
        obs = self.env.reset()

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

        robot0_base_id = self.env.sim.model.body_name2id("robot0_base")
        self.robot0_base_pos = self.env.sim.data.body_xpos[robot0_base_id]
        self.robot0_base_ori_rotm = self.env.sim.data.body_xmat[robot0_base_id].reshape((3,3)) # rotation matrix

        ###################################################

        self.cubeA_main_id = self.env.sim.model.body_name2id("cubeA_main")
        self.cubeB_main_id = self.env.sim.model.body_name2id("cubeB_main")

        pos_cubeA, rotm_cubeA = self.get_cube_pose(self.cubeA_main_id)
        pos_cubeB, rotm_cubeB = self.get_cube_pose(self.cubeB_main_id)

        self.pos_cubeA_GT = pos_cubeA
        self.pos_cubeB_GT = pos_cubeB

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

        self.rotm_cubeA_GT = rotm_cubeA
        self.rotm_cubeB_GT = rotm_cubeB

        if self.config['noisy_reference']:
            # print('noisy reference')
            rand_position1 = np.random.uniform(-0.007, 0.007, size=(3,))
            rand_position2 = np.random.uniform(-0.007, 0.007, size=(3,))
            # rand_position1 = np.array([0.005, -0.005, 0.005])
            self.pos_cubeA_noise = self.pos_cubeA_GT + rand_position1
            self.pos_cubeB_noise = self.pos_cubeB_GT + rand_position2
            # self.pos_cubeA_noise[2] = self.pos_cubeA_GT[2]
            # self.pos_cubeB_noise[2] = self.pos_cubeB_GT[2]

            angle = np.pi / 180 * 4
            rand_angle1 = np.random.uniform(-angle, angle, size = (3,))
            rand_angle2 = np.random.uniform(-angle, angle, size = (3,))
            # rand_angle1 = np.array([angle, -angle, angle])
            self.rotm_cubeA_noise = rotm_cubeA @ R.from_euler("xyz", rand_angle1).as_matrix()
            self.rotm_cubeB_noise = rotm_cubeB @ R.from_euler("xyz", rand_angle2).as_matrix()
        
        else:
            self.pos_cubeA_noise = self.pos_cubeA_GT
            self.pos_cubeB_noise = self.pos_cubeB_GT
            self.rotm_cubeA_noise = self.rotm_cubeA_GT
            self.rotm_cubeB_noise = self.rotm_cubeB_GT


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

            obs, reward, done, info = self.env.step(action)

        return obs

    def process_obs(self, obs):
        camera_names = self.config['camera_names']
        state_category = self.config['state_category']

        self.eef_pos = self.robot0_base_ori_rotm.T @ (obs['robot0_eef_pos'] - self.robot0_base_pos)
        self.eef_rotm = self.robot0_base_ori_rotm.T @ R.from_quat(obs['robot0_eef_quat_site']).as_matrix()

        self.eef_rotvec = R.from_matrix(self.eef_rotm).as_rotvec()

        name = "gripper0_right_grip_site"
        N_dof = self.env.robots[0].init_qpos.shape[0]
        
        jacp = self.env.sim.data.get_site_jacp(name)[:,:N_dof]
        jacr = self.env.sim.data.get_site_jacr(name)[:,:N_dof]

        J_full = np.zeros((6, N_dof))
        J_full[:3, :] = jacp
        J_full[3:, :] = jacr

        J_body = np.block([[self.eef_rotm.T, np.zeros((3, 3))], [np.zeros((3, 3)), self.eef_rotm.T]]) @ J_full

        eef_vel_body = J_body @ obs['robot0_joint_vel']

        eef_pose = np.concatenate((self.eef_pos, self.eef_rotvec))

        imgs = []
        for cam_name in camera_names:
            img = rearrange(obs[f"{cam_name}_image"], 'h w c -> c h w')
            imgs.append(img)
        
        img_stack = np.stack(imgs, axis=0)

        if state_category == "eef_pose_world":

            state = eef_pose

        elif state_category == "relative_init":
            # if self has init pos attribute
            if self.reset_flag:
                self.init_eef_pos = self.eef_pos
                self.init_eef_rotm = self.eef_rotm
                self.reset_flag = False
            
            g_i = np.eye(4)
            g_i[:3, 3] = self.init_eef_pos
            g_i[:3, :3] = self.init_eef_rotm

            g = np.eye(4)
            g[:3, 3] = self.eef_pos
            g[:3, :3] = self.eef_rotm

            g_rel = np.linalg.inv(g_i) @ g

            state = np.zeros(6)
            state[0:3] = g_rel[:3, 3]
            state[3:6] = R.from_matrix(g_rel[:3, :3]).as_rotvec()

        elif state_category == "GCEV":
            pd = self.pos_cubeA_noise
            Rd = self.rotm_cubeA_noise

            ep = self.eef_rotm.T @ (self.eef_pos - pd)
            eR = self.vee_map(Rd.T @ self.eef_rotm - self.eef_rotm.T @ Rd)

            state = np.concatenate((ep, eR))


        elif state_category == "eef_vel":
            state = eef_vel_body

        return state, img_stack

    def process_action(self, action):
        action_category = self.config['action_category']

        if action_category == "eef_pose_world":
            action_return = action

        elif action_category == "relative_fixed":
            g = np.eye(4)
            g[:3, 3] = self.eef_pos
            g[:3, :3] = self.eef_rotm

            g_actions_mat = np.zeros((action.shape[0], 4, 4))
            g_actions_mat[0, :4, :4] = g

            # action_cumsum = np.cumsum(action[:, :6], axis=0)
            # print("relative_fixed action shape", action.shape)

            for i in range(action.shape[0]):
                g_rel = np.eye(4)
                g_rel[:3, 3] = action[i, 0:3]
                g_rel[:3, :3] = R.from_rotvec(action[i, 3:6]).as_matrix()
                g_actions_mat[i, :4, :4] = g @ g_rel
                # g_actions_mat[i+1, :4, :4] = g_actions_mat[0, :4, :4] @ expm(hat_map(action_cumsum[i, :6]) * self.dt) 


            # print(f"g_actions_mat, determinants: {np.linalg.det(g_actions_mat[:, :3,:3])}")

            # return the action in SE(3) form
            return g_actions_mat
        
        elif action_category == "relative":
            g = np.eye(4)
            g[:3, 3] = self.eef_pos
            g[:3, :3] = self.eef_rotm

            g_rel = np.eye(4)
            g_rel[:3, 3] = action[0:3]
            g_rel[:3, :3] = R.from_rotvec(action[3:6]).as_matrix()

            g_a = g @ g_rel

            action_return = np.zeros(7)
            action_return[0:3] = g_a[:3, 3]
            action_return[3:6] = R.from_matrix(g_a[:3, :3]).as_rotvec()
            # gripper action
            action_return[-1] = action[-1]

        elif action_category == "desired_frame" or action_category == "desired_frame_correction":


            # cubeA_pos = self.pos_cubeA_GT
            # cubeA_rotm = self.rotm_cubeA_GT

            cubeA_pos = self.pos_cubeA_noise
            cubeA_rotm = self.rotm_cubeA_noise

            gd = np.eye(4)
            gd[:3, 3] = cubeA_pos
            gd[:3, :3] = cubeA_rotm

            # print(f"cubeA_pos: {cubeA_pos}, cubeA_rotm: {cubeA_rotm}")

            g_rel = np.eye(4)
            g_rel[:3, 3] = action[0:3]
            g_rel[:3, :3] = R.from_rotvec(action[3:6]).as_matrix()

            g_a = gd @ g_rel

            action_return = np.zeros(7)
            action_return[0:3] = g_a[:3, 3]
            action_return[3:6] = R.from_matrix(g_a[:3, :3]).as_rotvec()
            # gripper action
            action_return[-1] = action[-1]

        elif action_category == "relative_init":
            g_i = np.eye(4)
            g_i[:3, 3] = self.init_eef_pos
            g_i[:3, :3] = self.init_eef_rotm

            g_rel = np.eye(4)
            g_rel[:3, 3] = action[0:3]
            g_rel[:3, :3] = R.from_rotvec(action[3:6]).as_matrix()

            g_a = g_i @ g_rel

            action_return = np.zeros(7)
            action_return[0:3] = g_a[:3, 3]
            action_return[3:6] = R.from_matrix(g_a[:3, :3]).as_rotvec()

            # gripper action
            action_return[-1] = action[-1]

        return action_return

    def eval_bc(self):
        state_dim = self.config['state_dim']
        action_dim = self.config['action_dim']
        policy_config = self.config['policy_config']
        camera_names = self.config['camera_names']
        max_timesteps = self.config['episode_len'] + 100
        # temporal_agg = config['temporal_agg']

        print(f"state_category: {self.config['state_category']}, action_category: {self.config['action_category']}")

        temporal_agg = True
        query_frequency = 1
        num_queries = policy_config['num_queries']

        max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

        self.state_category = self.config['state_category']
        self.action_category = self.config['action_category']

        num_rollouts = self.eval_config['num_episodes']

        success_info = {
            "success": [],
            "pos_error": [],
            "rot_error": [],
        }

        for rollout_id in range(num_rollouts):
            np.random.seed(rollout_id+ 50)
            obs = self.reset_env()

            ### evaluation loop
            if temporal_agg:
                if self.action_category == "relative_fixed":
                    all_time_actions_np = np.zeros([max_timesteps, max_timesteps+num_queries, 4,4])
                    all_time_grasp_np = np.zeros([max_timesteps, max_timesteps+num_queries, 1])

                else:
                    all_time_actions = np.zeros([max_timesteps, max_timesteps+num_queries, action_dim])

            state_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
            image_list = [] # for visualization
            state_list = []
            action_list = []
            success_list = []
            with torch.inference_mode():
                for t in range(max_timesteps):

                    # process observations
                    state_numpy, curr_image = self.process_obs(obs)

                    # # show curr image
                    # plt.figure()
                    # plt.imshow(curr_image[0].transpose(1, 2, 0))
                    # plt.show()
                    
                    state = self.pre_process(state_numpy)
                    state = torch.from_numpy(state).float().cuda().unsqueeze(0)
                    state_history[:, t] = state
                    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

                    ### query policy
                    if self.action_category == "relative_fixed":
                        if self.config['policy_class'] == "ACT":
                            if t % query_frequency == 0:
                                all_actions = self.policy(state, curr_image).squeeze(0).cpu().numpy()
                                unnormalized_actions = self.post_process(all_actions)
                                all_actions_pose = self.process_action(unnormalized_actions)                                
                                
                            if temporal_agg:  
                                all_time_actions_np[[t], t:t+num_queries] = all_actions_pose
                                actions_for_curr_step = all_time_actions_np[:, t]

                                # print(np.linalg.det(actions_for_curr_step[:,:3,:3]))

                                actions_populated = (np.linalg.det(actions_for_curr_step[:,:3,:3]) != 0)
                                # print(actions_for_curr_step.shape)
                                actions_for_curr_step = actions_for_curr_step[actions_populated]

                                #TODO I need to fix this
                                # print(actions_for_curr_step.shape, actions_for_curr_step.reshape((-1, 4, 4)).shape)

                                all_time_grasp_np[[t], t:t+num_queries] = unnormalized_actions[:, -1].reshape((-1,1))
                                # print(unnormalized_actions[:, -1].shape)
                                grasp_for_curr_step = all_time_grasp_np[:, t]
                                grasp_populated = (grasp_for_curr_step != 0)
                                grasp_for_curr_step = grasp_for_curr_step[grasp_populated]
                                
                                # print(actions_populated.sum(), grasp_populated.sum())

                                k = 0.01
                                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                                exp_weights = exp_weights / exp_weights.sum()
                                
                                translation = (actions_for_curr_step[:, :3, 3] * exp_weights.reshape((-1,1))).sum(axis=0, keepdims=True)
                                rotation_object = R.from_matrix(actions_for_curr_step[:, :3, :3])
                                rotation = rotation_object.mean(weights = exp_weights)

                                grasp = (grasp_for_curr_step * exp_weights).sum(axis=0, keepdims=True)

                                # print(grasp.shape)

                                action = np.zeros((7,))
                                action[0:3] = translation
                                action[3:6] = rotation.as_rotvec()
                                action[-1] = grasp[0]                              
                            else:
                                action_pose = all_actions_pose[:, t % query_frequency]
                                action = np.zeros((7,))
                                action[0:3] = action_pose[:3,3]
                                action[3:6] = R.from_matrix(action_pose[:3,:3]).as_rotvec()
                                action[-1] = unnormalized_actions[:, t % query_frequency][-1]
                                
                        elif self.config['policy_class'] == "CNNMLP":
                            raw_action = self.policy(state, curr_image)
                        else:
                            raise NotImplementedError
                        
                    else:
                        if self.config['policy_class'] == "ACT":
                            if t % query_frequency == 0:
                                all_actions = self.policy(state, curr_image)
                                all_actions = all_actions.squeeze(0).cpu().numpy()
                                unnormalized_actions = self.post_process(all_actions)
                            if temporal_agg:
                                # print("using temporal aggregation")
                                all_time_actions[[t], t:t+num_queries] = unnormalized_actions
                                actions_for_curr_step = all_time_actions[:, t]
                                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                                actions_for_curr_step = actions_for_curr_step[actions_populated]
                                k = 0.01
                                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                                exp_weights = exp_weights / exp_weights.sum()
                                exp_weights = exp_weights.reshape((-1,1))
                                action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                            else:
                                action = unnormalized_actions[:, t % query_frequency]
                        elif self.config['policy_class'] == "CNNMLP":
                            raw_action = self.policy(state, curr_image)
                            action = self.post_process(raw_action.squeeze(0).cpu.numpy())
                        else:
                            raise NotImplementedError

                        ### post-process actions
                        # raw_action = raw_action.squeeze(0).cpu().numpy()
                        # action = self.post_process(raw_action)

                        if action.shape == (1, action_dim):
                            action = action.squeeze(0)

                        action = self.process_action(action)
                    
                    ### step the environment
                    obs, reward, done, info = self.env.step(action)
                    # plt.show()

                    ### for visualization
                    state_list.append(state_numpy)
                    action_list.append(action)
            
            #update position of the block

            pos_cubeA, rotm_cubeA = self.get_cube_pose(self.cubeA_main_id)
            state_numpy = self.process_obs(obs) # update self.eef_pos and self.eef_rotm
            if self.env_config['env_name'] == "StackCustom":
                if self.eval_config["stage"] == 0:
                    # print(self.eef_pos, pos_cubeA, self.pos_cubeA_GT, rotm_cubeA, self.rotm_cubeA_GT)                    
                    if np.linalg.norm(self.eef_pos - pos_cubeA) < 0.02 and abs(self.eef_pos[2] - pos_cubeA[2]) < 0.01:
                        success_info["success"].append(1)
                    else:
                        success_info["success"].append(0)

                    success_info["pos_error"].append(self.eef_pos - self.pos_cubeA_GT)
                    error_euler = R.from_matrix(self.rotm_cubeA_GT.T @ self.eef_rotm).as_euler("xyz", degrees=True)
                    success_info["rot_error"].append(error_euler)

            print(f"Rollout {rollout_id} done, success: {success_info['success'][-1]}, pos_error: {success_info['pos_error'][-1]}, rot_error: {success_info['rot_error'][-1]}")
            # save success_info at ckpt directory
        success_info_path = os.path.join(self.config['ckpt_dir'], f'success_info.pkl')
        with open(success_info_path, 'wb') as f:
            pickle.dump(success_info, f)
        print(f"success_rate: {sum(success_info['success'])} / {len(success_info['success'])}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir : disabled right now')
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

    EE = EvalEnv(vars(parser.parse_args()))
    
    EE.eval_bc()
