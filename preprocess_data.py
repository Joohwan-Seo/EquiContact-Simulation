import numpy as np
import argparse
import yaml
import pickle

from scipy.spatial.transform import Rotation as R

import h5py
# import IPython

import time

import os

from utils.transform_utils import *

def main(args):
    # Get config
    with open(args['config_file'], 'r') as f:
        config = yaml.safe_load(f)
    env_config = config['env_parameters']
    task_config = config['task_parameters']

    num_episodes = task_config['num_episodes']
    dataset_dir = task_config['raw_dataset_dir']
    processed_dataset_dir = task_config['processed_dataset_dir']
    max_timesteps = task_config['episode_len']
    state_category = task_config['state_category']
    action_category = task_config['action_category']
    noisy_reference = task_config['noisy_reference']
    noisy_reference_type = task_config['noisy_reference_type']

    num_queries = task_config['num_queries']
    state_dim = task_config['state_dim']
    action_dim = task_config['action_dim']

    target_stage = task_config['stage']

    camera_names = task_config['camera_names']

    for episode_idx in range(num_episodes):
        with open(f"{dataset_dir}/demo_{episode_idx+1}.pkl", 'rb') as f:
            rollouts = pickle.load(f)

        data_dict = {
            '/observations/state': [],
            '/action': [],
            '/is_padded': []
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # Get the indices of the frames where the stage is target_stage
        if target_stage == ("whole" or "Whole"):
            # print("whole")
            target_stage = stages
            idx = [i for i, stage in enumerate(stages) if stage == target_stage[i]]
        else:
            target_stage = int(target_stage)
            idx = []
            for i in range(len(stages)):
                if stages[i] == target_stage:
                    idx.append(i)

        ep_len = len(idx)

        obs_data = rollouts['observations'][:ep_len]
        action_data = rollouts['actions'][:ep_len]
        stages = rollouts['stage'][:ep_len]
        target_stage = task_config['stage']

        padding_mask = np.zeros((max_timesteps,), dtype=bool)
        padding_mask[:ep_len] = True
        action_data += [action_data[-1]] * (num_queries)

        cubeA_pos = obs_data[0]['cubeA_pos']
        cubeB_pos = obs_data[0]['cubeB_pos']

        cubeA_rotm = R.from_quat(obs_data[0]['cubeA_quat']).as_matrix()
        cubeB_rotm = R.from_quat(obs_data[0]['cubeB_quat']).as_matrix()

        cubeA_pos_GT = cubeA_pos.copy()
        cubeB_pos_GT = cubeB_pos.copy()

        cubeA_rotm_GT = cubeA_rotm.copy()
        cubeB_rotm_GT = cubeB_rotm.copy()

        if noisy_reference:
            if noisy_reference_type == "fixed": # noisify reference
                pos_noise = np.random.uniform(-1, 1, 3) * 0.04
                angle_noise = np.random.uniform(-1, 1, 3) * 24 / 180 * np.pi
                cubeA_pos += pos_noise
                cubeA_rotm = cubeA_rotm @ R.from_euler('xyz', angle_noise).as_matrix()
                cubeB_pos += pos_noise
                cubeB_rotm = cubeB_rotm @ R.from_euler('xyz', angle_noise).as_matrix()
        


        # print(len(idx))

        poses_dict_buffer = {
            "pos": [],
            "rotm": []
        }

        for j in range(ep_len):
            if j not in idx:
                continue

            # Proprioceptive observations
            eef_pos = obs_data[j]['robot0_eef_pos'] 
            rotm = R.from_quat(obs_data[j]['robot0_eef_quat_site']).as_matrix()
            eef_vel = obs_data[j]['robot0_eef_vel_body']

            poses_dict_buffer["pos"].append(eef_pos)
            poses_dict_buffer["rotm"].append(rotm)

            # if stages[j] == 0: use cubeA as a reference (noisy reference)
            if stages[j] == 0:
                pd = cubeA_pos
                Rd = cubeA_rotm
                pd_GT = cubeA_pos_GT
                Rd_GT = cubeA_rotm_GT

            else:
                pd = cubeB_pos
                Rd = cubeB_rotm
                pd_GT = cubeB_pos_GT
                Rd_GT = cubeB_rotm_GT

            states = []
            actions = []

            for state in state_category:
                if state == "world_pose":
                    state = np.concatenate((eef_pos, rotm_to_rot6d(rotm)))
                elif state == "eef_vel":
                    state = eef_vel
                elif state == "GCEV":
                    if noisy_reference:
                        if noisy_reference_type == "random":
                            pos_noise = np.random.uniform(-1, 1, 3) * 0.04
                            angle_noise = np.random.uniform(-1, 1, 3) * 24 / 180 * np.pi

                            pd = pd_GT + pos_noise
                            Rd = Rd_GT @ R.from_euler('xyz', angle_noise).as_matrix()                    
                    ep = rotm.T @ (eef_pos - pd)
                    eR = vee_map(Rd.T @ rotm - rotm.T @ Rd)

                    state = np.concatenate((ep, eR))

                states.append(state)



            for action in action_category:
                if action == "world_pose":
                    action = np.zeros((num_queries, 9))
                    for k in range(num_queries):
                        action[k, :3] = action_data[j+k][:3]
                        action[k, 3:] = rotvec_to_rot6d(action_data[j+k][3:6])

                elif action == "relative":
                    action = np.zeros((num_queries, 6))
                    for k in range(num_queries):
                        g_a = np.eye(4)
                        g_a[:3,3] = action_data[j+k][:3]
                        g_a[:3,:3] = R.from_rotvec(action_data[j+k][3:6]).as_matrix()

                        g = np.eye(4)
                        g[:3,3] = eef_pos
                        g[:3,:3] = rotm
                        g_rel = np.linalg.inv(g) @ g_a
                        action[k, :3] = g_rel[:3, 3]
                        action[k, 3:] = R.from_matrix(g_rel[:3, :3]).as_rotvec()
                
                elif action == "gripper":
                    action = np.zeros((num_queries, 1))
                    for k in range(num_queries):
                        action[k, 0] = action_data[j+k][6]
                    
                actions.append(action)

            processed_states = np.concatenate(states, axis=0)
            assert processed_states.shape == (state_dim,), f"State shape mismatch: {processed_states.shape} != {(state_dim,)}"

            processed_action = np.concatenate(actions, axis=1)
            assert processed_action.shape == (num_queries, action_dim), f"Action shape mismatch: {processed_action.shape} != {(num_queries, action_dim)}"

            current_padding_mask = padding_mask[j:j + num_queries]

            data_dict['/observations/state'].append(processed_states)
            data_dict['/action'].append(processed_action)
            data_dict['/is_padded'].append(current_padding_mask)

            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(obs[j][f"{cam_name}_image"])

        for key, seq in data_dict.items():
            while len(seq) < ep_len:
                seq.append(seq[-1])

        tic = time.time()
 
        dir = processed_dataset_dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        dataset_path = dir + f"/episode_{episode_idx}"
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 256, 256, 3), dtype='uint8',
                                         chunks=(1, 256, 256, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            state_ = obs.create_dataset('state', (max_timesteps, state_dim))
            action_ = root.create_dataset('action', (max_timesteps, num_queries, action_dim))
            is_padded = root.create_dataset('is_padded', (max_timesteps, num_queries), dtype=bool)

            for name, array in data_dict.items():
                root[name][...] = array       
        print(f"Episode {episode_idx} saved to {dataset_path}.hdf5 in {time.time()-tic:.2f} seconds")
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ACT")
    parser.add_argument("--config_file", type=str, default="config/train/ACT_stack_08.yaml")

    args = parser.parse_args()

    main(vars(args))