import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import mujoco

import robosuite as suite
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

import copy
from utils.transform_utils import SE3_log_map, SE3_exp_map

import matplotlib.pyplot as plt

class PolicyPlayerPIH:
    def __init__ (self, env, render= True, save = True, randomized = True, debug = False):
        '''
        Playing of scripted policy for the two-arm handover role environment
        env: TwoArmHandoverRole environment
        render: bool, whether to render the environment

        NOTE: The waypoints are hardcoded in the setup_waypoints function
        NOTE: Observations are modified to be the "local"
        For example, obs['robot0_eef_pos'] is the position of the robot0 end-effector in the robot0 base frame

        NOTE outputs only rollout, composed of 
        rollout["observations"] = [obs0, obs1, obs2, ...]
        obs0: dictionary with observation keys
        rollout["actions"] = [action0, action1, action2, ...] 
        action0: numpy array of shape (14,) with the action for robot0 and robot1

        Now the preprocessing on the observation is not needed.
        '''

        #TODO(JS) Update the Policy player so that the actions are in the P-control action towards the waypoints
        
        
        self.env = env

        self.control_freq = env.control_freq
        self.dt = 1.0 / self.control_freq
        self.max_time = 10
        self.max_steps = int(self.max_time / self.dt)

        self.render = render
        self.save = save
        self.randomized = randomized
        self.debug = debug

        # robot0_base_body_id = env.sim.model.body_name2id("robot0:base")
        # possible: 'robot0_base', 'robot0_fixed_base_link', 'robot0_shoulder_link'

        # Extract the base position and orientation (quaternion) from the simulation data
        robot0_base_body_id = self.env.sim.model.body_name2id("robot0_base")
        self.robot0_base_pos = self.env.sim.data.body_xpos[robot0_base_body_id]
        self.robot0_base_ori_rotm = self.env.sim.data.body_xmat[robot0_base_body_id].reshape((3,3))

        # Rotation matrix of robots for the home position, both in their own base frame
        self.R_be_home = np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, -1]])

        self.n_action = self.env.action_spec[0].shape[0]

        self.max_step_move = int(5 * self.control_freq) # 10 seconds
        self.max_step_grasp = int(0.5 * self.control_freq)

        # Last thing to do 
        obs = self.reset()

        if self.debug:
            print("Possible keys in obs:", obs.keys())

    def get_cube_pose(self, cube_id):
        pos_cube_world = self.env.sim.data.body_xpos[cube_id]
        rotm_cube_world = self.env.sim.data.body_xmat[cube_id].reshape((3,3)) # rotation matrix

        pos_cube = self.robot0_base_ori_rotm.T @ (pos_cube_world - self.robot0_base_pos)
        rotm_cube = self.robot0_base_ori_rotm.T @ rotm_cube_world

        return pos_cube, rotm_cube


    def reset(self, seed = 0):
        """
        Resetting environment. Re-initializing the waypoints too.
        """
        np.random.seed(seed)
        obs = self.env.reset()

        self.rollout = {}
        self.rollout["observations"] = []
        self.rollout["actions"] = []

        """
        Available "body" names = ('world', 'table', 'left_eef_target', 'right_eef_target', 'robot0_base', 
        'robot0_link0', 'robot0_link1', 'robot0_link2', 'robot0_link3', 'robot0_link4', 'robot0_link5', 'robot0_link6', 
        'robot0_right_hand', 'gripper0_right_right_gripper', 'gripper0_right_eef', 'gripper0_right_gripper_body', 'gripper0_right_leftfinger', 'gripper0_right_finger_joint1_tip', 'gripper0_right_rightfinger', 'gripper0_right_finger_joint2_tip', 
        'fixed_mount0_base', 'fixed_mount0_controller_box', 'fixed_mount0_pedestal_feet', 'fixed_mount0_torso', 'fixed_mount0_pedestal', 
        'peg_main', 'hole_main'
        """
        peg_id = self.env.sim.model.site_name2id("peg_grip_site")
        pos_peg_world = self.env.sim.data.site_xpos[peg_id]
        rotm_peg_world = self.env.sim.data.site_xmat[peg_id].reshape((3,3)) # rotation matrix

        self.pos_peg = self.robot0_base_ori_rotm.T @ (pos_peg_world - self.robot0_base_pos)
        self.rotm_peg = self.robot0_base_ori_rotm.T @ rotm_peg_world

        hole_id = self.env.sim.model.site_name2id("hole_default_site")
        pos_hole_world = self.env.sim.data.site_xpos[hole_id]
        rotm_hole_world = self.env.sim.data.site_xmat[hole_id].reshape((3,3)) # rotation matrix

        self.pos_hole = self.robot0_base_ori_rotm.T @ (pos_hole_world - self.robot0_base_pos)
        self.rotm_hole = self.robot0_base_ori_rotm.T @ rotm_hole_world


        if self.randomized:
            rand_xy_A = np.random.uniform(-0.003, 0.003, size=(2,))
            rand_xy_B = np.random.uniform(-0.01, 0.01, size=(2,))
            rand_z_A = np.random.uniform(-0.005, 0)
        else:
            rand_xy_A = np.array([0, 0])
            rand_xy_B = np.array([0, 0])
            rand_z_A = 0
        self.pos_peg = self.pos_peg + self.rotm_peg @ np.array([0, 0, 0]) + np.array([rand_xy_A[0], rand_xy_A[1], rand_z_A - 0.005])
        self.pos_hole = self.pos_hole + self.rotm_hole @ np.array([0, 0, 0]) + np.array([rand_xy_B[0], rand_xy_B[1], 0])

        #Rotation matrix post processing for cube A and cube B
        rotm_x = R.from_euler("x", np.pi).as_matrix()
        self.rotm_peg = self.rotm_peg @ rotm_x
        self.rotm_hole = self.rotm_hole @ rotm_x 

        # print("peg and hole position:", self.pos_peg, self.pos_hole)
        # print("peg and hole rotation:", self.rotm_peg, self.rotm_hole)

        # rotate rotm_cubeA and rotm_cubeB with 90 degrees interval, choose the rotation matrix has a minmum distance to the R_be_home
        rotm_list_peg = []
        rotm_list_hole = []
        for i in range(4):
            rotm_list_peg.append(self.rotm_peg @ R.from_euler("z", i * np.pi / 2).as_matrix())
            rotm_list_hole.append(self.rotm_hole @ R.from_euler("z", i * np.pi / 2).as_matrix())
        
        # find the rotation matrix that has a minimum distance to the R_be_home
        min_dist_peg = 100 # any large number bigger than 4
        min_dist_hole = 100

        for i in range(4):
            dist_peg = np.trace(np.eye(3) - self.R_be_home.T @ rotm_list_peg[i])
            dist_hole = np.trace(np.eye(3) - self.R_be_home.T @ rotm_list_hole[i])

            if dist_peg < min_dist_peg:
                min_dist_peg = dist_peg
                self.rotm_peg = rotm_list_peg[i]

            if dist_hole < min_dist_hole:
                min_dist_hole = dist_hole
                self.rotm_hole = rotm_list_hole[i]

        rand_init_pos = np.random.uniform(-0.08, 0.08, size=(3,))
        init_pos = self.pos_peg + np.array([rand_init_pos[0], rand_init_pos[1], 0.1])

        rand_init_euler = np.random.uniform(-10/180*np.pi, 10/180*np.pi, size=(3,))
        init_rotm = self.rotm_peg @ R.from_euler("xyz", rand_init_euler).as_matrix()

        arrived = False
        for i in range(100):
            if arrived:
                break
            action = np.zeros(int(self.n_action))
            action[0:3] = init_pos
            action[3:6] = R.from_matrix(init_rotm).as_rotvec()
            action[6] = -1
            pos, rotm = self.get_poses(obs)
            arrived = self.check_arrived(pos, rotm, init_pos, init_rotm, 0.001)

            obs, reward, done, info = self.env.step(action)

        self.setup_waypoints()

        return obs

    def setup_waypoints(self):
        self.waypoints = []

        # Go up to the peg, wp0
        waypoint = {
            "pos": self.pos_peg + np.array([0, 0, 0.1]),
            "rotm": self.rotm_peg,
            "gripper": -1,
            "property": "move",
            "threshold": 0.01
        }

        self.waypoints.append(waypoint)

        # Go down to the peg, wp1 # already to the grip site
        waypoint = {
            "pos": self.pos_peg,
            "rotm": self.rotm_peg,
            "gripper": -1,
            "property": "move",
            "threshold": 0.003
        }

        self.waypoints.append(waypoint)

        # Grasp the peg, wp2
        waypoint = {
            "pos": self.pos_peg,
            "rotm": self.rotm_peg,
            "gripper": 1,
            "property": "grasp",
            "threshold": 0.003
        }

        self.waypoints.append(waypoint)

        # Go up to the peg, wp3
        waypoint = {
            "pos": self.pos_peg + np.array([0, 0, 0.1]),
            "rotm": self.rotm_peg,
            "gripper": 1,
            "property": "move",
            "threshold": 0.01
        }

        self.waypoints.append(waypoint)

        # Go up to the hole, wp4
        waypoint = {
            "pos": self.pos_hole + np.array([0, 0, 0.15]),
            "rotm": self.rotm_hole,
            "gripper": 1,
            "property": "move",
            "threshold": 0.01
        }

        self.waypoints.append(waypoint)

        # Go up to the hole precisely, wp5
        waypoint = {
            "pos": self.pos_hole + np.array([0, 0, 0.13]),
            "rotm": self.rotm_hole,
            "gripper": 1,
            "property": "move",
            "threshold": 0.0001
        }

        self.waypoints.append(waypoint)

        waypoint = {
            "pos": self.pos_hole + np.array([0, 0, 0.13]),
            "rotm": self.rotm_hole,
            "gripper": 1,
            "property": "move",
            "threshold": 0.0001
        }

        self.waypoints.append(waypoint)

        # Put the peg into the hole, wp6
        waypoint = {
            "pos": self.pos_hole + np.array([0, 0, 0.07]),
            "rotm": self.rotm_hole,
            "gripper": 1,
            "property": "move",
            "threshold": 0.0001
        }

        self.waypoints.append(waypoint)

        # Release the peg, wp6
        waypoint = {
            "pos": self.pos_hole,
            "rotm": self.rotm_hole,
            "gripper": -1,
            "property": "grasp",
            "threshold": 0.003
        }


        
        
    #alpha = 0.75 was working
    def convert_action_robot(self, robot_pos, robot_rotm, robot_goal_pos, robot_goal_rotm, robot_gripper, alpha = 0.5):
        action = np.zeros(int(self.n_action))

        g = np.eye(4)
        g[0:3, 0:3] = robot_rotm
        g[0:3, 3] = robot_pos

        gd = np.eye(4)
        gd[0:3, 0:3] = robot_goal_rotm
        gd[0:3, 3] = robot_goal_pos

        xi = SE3_log_map(np.linalg.inv(g) @ gd)

        gd_modified = g @ SE3_exp_map(alpha * xi)

        action[0:3] = gd_modified[:3,3]
        action[3:6] = R.from_matrix(gd_modified[:3,:3]).as_rotvec()
        action[6] = robot_gripper

        return action
    
    
    def get_poses(self, obs):
        robot0_pos_world = obs['robot0_eef_pos']
        robot0_rotm_world = R.from_quat(obs['robot0_eef_quat_site']).as_matrix()

        robot0_pos = self.robot0_base_ori_rotm.T @ (robot0_pos_world - self.robot0_base_pos)
        robot0_rotm = self.robot0_base_ori_rotm.T @ robot0_rotm_world
        
        return robot0_pos, robot0_rotm
    
    def check_arrived(self, pos1, rotm1, pos2, rotm2, threshold = 0.05):
        pos_diff = pos1 - pos2
        rotm_diff = rotm2.T @ rotm1

        #SE(3) distance proposed by Seo et al. 2023
        distance = np.sqrt(0.5 * np.linalg.norm(pos_diff)**2 + np.trace(np.eye(3) - rotm_diff))

        if distance < threshold:
            return True
        else:
            return False
        
    def get_Jacobian(self):
        """
        Jacobian of the robot0 end-effector
        """
        name = "gripper0_right_grip_site"
        N_dof = self.env.robots[0].init_qpos.shape[0]
        
        jacp = self.env.sim.data.get_site_jacp(name)[:,:N_dof]
        jacr = self.env.sim.data.get_site_jacr(name)[:,:N_dof]

        # print(jacp.shape, jacr.shape)

        J_full = np.zeros((6, N_dof))
        J_full[:3, :] = jacp
        J_full[3:, :] = jacr

        return J_full
        
    def execute_waypoint(self, waypoint, last_obs):
        obs = last_obs
        robot_arrived = False
        if waypoint["property"] == "move":
            for i in range(self.max_step_move):
                pos, rotm = self.get_poses(obs)

                if not robot_arrived:
                    goal_pos = waypoint['pos']
                    goal_rotm = waypoint['rotm']
                    action = self.convert_action_robot(pos, rotm, goal_pos, goal_rotm, waypoint['gripper'])
                    robot_arrived = self.check_arrived(pos, rotm, goal_pos, goal_rotm, waypoint['threshold'])

                else:
                    break

                obs, reward, done, info = self.env.step(action)

                if self.debug:
                    # if i % 10 == 0:
                    print(action)
                
                if self.render:
                    self.env.render()

                if self.save:
                    self.rollout["observations"].append(self.process_obs(obs))
                    self.rollout["actions"].append(action)
        
        elif waypoint["property"] == "grasp":
            for i in range(self.max_step_grasp):
                pos, rotm = self.get_poses(obs)

                goal_pos = waypoint['pos']
                goal_rotm = waypoint['rotm']

                action = self.convert_action_robot(pos, rotm, goal_pos, goal_rotm, waypoint['gripper'])

                obs, reward, done, info = self.env.step(action)

                if self.render:
                    self.env.render()

                if self.save:
                    self.rollout["observations"].append(self.process_obs(obs))
                    self.rollout["actions"].append(action)

        last_obs = obs
        return last_obs

    
    def get_demo(self, seed):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed)
        # print("hammer length:", self.env.hammer.handle_length)
        # print("distance between tables:", self.robot0_base_pos - self.robot1_base_pos)

        robot_arrived = False

        last_obs = obs

        for waypoint in self.waypoints:
            last_obs_update = self.execute_waypoint(waypoint, last_obs)
            last_obs = last_obs_update
        
        cubeA_pos, cubeA_rotm = self.get_cube_pose(self.env.sim.model.body_name2id("cubeA_main"))
        cubeB_pos, cubeB_rotm = self.get_cube_pose(self.env.sim.model.body_name2id("cubeB_main"))

        if cubeA_pos[2] - cubeB_pos[2] > 0:
            if np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2]) < 0.02:
                done = True

        else:
            done = False

        return self.rollout, done

    def process_obs(self, obs):

        processed_obs = copy.deepcopy(obs)
        pos, rotm= self.get_poses(obs)

        # world to body
        processed_obs['robot0_eef_pos'] = self.robot0_base_ori_rotm.T @ (pos - self.robot0_base_pos)
        processed_obs['robot0_eef_quat_site'] = R.from_matrix(self.robot0_base_ori_rotm.T @ rotm).as_quat()

        eef_rotm = R.from_quat(obs['robot0_eef_quat_site']).as_matrix()

        J_full = self.get_Jacobian()

        J_body = np.block([[eef_rotm.T, np.zeros((3, 3))], [np.zeros((3, 3)), eef_rotm.T]]) @ J_full

        eef_vel_body = J_body @ obs['robot0_joint_vel']

        processed_obs['robot0_eef_vel_body'] = eef_vel_body

        return processed_obs

    def test(self):
        """
        Testing the environment during the development of the scripted policy
        """

        obs = self.reset()
        goal_pos = np.array([0.5, 0.1, 0.2])
        goal_rotm = self.rotm_cubeB

        print(obs.keys())

        for i in range(self.max_steps):
            pos, rotm = self.get_poses(obs)

            action = self.convert_action_robot(pos, rotm, goal_pos, goal_rotm, -1) #(7,)

            action[0:3] = goal_pos
            action[3:6] = R.from_matrix(goal_rotm).as_rotvec()
            action[6] = 1

            # print("position error:", pos - goal_pos)
            # print("rotation error:", np.trace(np.eye(3) - goal_rotm.T @ rotm))
            # print("joint angles:", obs['robot0_joint_pos'])
            # print("gripper qpos:", obs['robot0_gripper_qpos'])

            obs, reward, done, info = self.env.step(action)

            obs = self.process_obs(obs)

            print(obs['robot0_eef_vel_body'])
            self.env.render()

            # print(obs['robot0_gripper_qpos'])
            