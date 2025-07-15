import numpy as np
import argparse
import yaml
import pickle

from scipy.spatial.transform import Rotation as R

import h5py
# import IPython

import time

import os

from utils.transform_utils import hat_map, vee_map

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

    target_stage = task_config['stage']

    camera_names = task_config['camera_names']

    for episode_idx in range(num_episodes):
        with open(f"{dataset_dir}/demo_{episode_idx+1}.pkl", 'rb') as f:
            rollouts = pickle.load(f)

        data_dict = {
            '/observations/state': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        obs = rollouts['observations']
        actions = rollouts['actions']
        stages = rollouts['stage']
        target_stage = task_config['stage']

        # print(target_stage)

        if target_stage == ("whole" or "Whole"):
            # print("whole")
            target_stage = stages
            idx = [i for i, stage in enumerate(stages) if stage == target_stage[i]]
        else:
            target_stage = int(target_stage)
            idx = []
            for i in range(len(stages)):
                if stages[i] == target_stage:
                    # print(stages[i], target_stage)
                    idx.append(i)

        # Get the indices of the frames where the stage is 1

        cubeA_pos = obs[0]['cubeA_pos']
        cubeB_pos = obs[0]['cubeB_pos']

        cubeA_rotm = R.from_quat(obs[0]['cubeA_quat']).as_matrix()
        cubeB_rotm = R.from_quat(obs[0]['cubeB_quat']).as_matrix()

        cubeA_pos_GT = cubeA_pos.copy()
        cubeB_pos_GT = cubeB_pos.copy()

        cubeA_rotm_GT = cubeA_rotm.copy()
        cubeB_rotm_GT = cubeB_rotm.copy()

        if noisy_reference: # noisify reference
            pos_noise = np.random.uniform(-1, 1, 3) * 0.01
            angle_noise = np.random.uniform(-1, 1, 3) * 6 / 180 * np.pi
            cubeA_pos += pos_noise
            cubeA_rotm = cubeA_rotm @ R.from_euler('xyz', angle_noise).as_matrix()
        

        ep_len = len(idx)
        # print(len(idx))

        init_eef_pos = obs[0]['robot0_eef_pos']
        init_eef_rotm = R.from_quat(obs[0]['robot0_eef_quat_site']).as_matrix()

        for j in range(ep_len):
            if j not in idx:
                continue

            # Proprioceptive observations
            eef_pos = obs[j]['robot0_eef_pos'] 
            rotm = R.from_quat(obs[j]['robot0_eef_quat_site']).as_matrix()
            eef_vel = obs[j]['robot0_eef_vel_body']


            if stages[j] == 0:
                pd = cubeA_pos
                Rd = cubeA_rotm
            else:
                pd = cubeB_pos
                Rd = cubeB_rotm
            
            
            if state_category == "eef_pose_world":
                eef_rotvec = R.from_matrix(rotm).as_rotvec()
                eef_pose = np.concatenate((eef_pos, eef_rotvec))
                state = eef_pose

            elif state_category == "eef_vel":
                state = eef_vel

            elif state_category == "GCEV":
                if noisy_reference:
                    if noisy_reference_type == "random":
                        pos_noise = np.random.uniform(-1, 1, 3) * 0.02
                        angle_noise = np.random.uniform(-1, 1, 3) * 12 / 180 * np.pi
                        cubeA_pos = cubeA_pos_GT + pos_noise
                        cubeA_rotm = cubeA_rotm_GT @ R.from_euler('xyz', angle_noise).as_matrix()

                        pd = cubeA_pos
                        Rd = cubeA_rotm

                    elif noisy_reference_type == "fixed":
                        pass

                    else:
                        raise ValueError("noisy_reference_type must be 'random' or 'fixed'")
                else:
                    pass
                
                ep = rotm.T @ (eef_pos - pd)
                eR = vee_map(Rd.T @ rotm - rotm.T @ Rd)

                state = np.concatenate((ep, eR))

            elif state_category == "relative_init":
                g_i = np.eye(4)
                g_i[:3, 3] = init_eef_pos
                g_i[:3, :3] = init_eef_rotm

                g = np.eye(4)
                g[:3, 3] = eef_pos
                g[:3, :3] = rotm

                g_rel = np.linalg.inv(g_i) @ g

                state = np.zeros(6)
                state[0:3] = g_rel[:3, 3]
                state[3:6] = R.from_matrix(g_rel[:3, :3]).as_rotvec()


            if action_category == "eef_pose_world":
                action = actions[j]

            elif action_category == "relative":
                g = np.eye(4)
                g[:3, 3] = eef_pos
                g[:3, :3] = rotm

                g_a = np.eye(4)
                g_a[:3, 3] = actions[j][:3]
                g_a[:3, :3] = R.from_rotvec(actions[j][3:6]).as_matrix()

                g_rel = np.linalg.inv(g) @ g_a

                action = np.zeros(7)
                action[0:3] = g_rel[:3, 3]
                action[3:6] = R.from_matrix(g_rel[:3, :3]).as_rotvec()

                # gripper action
                action[-1] = actions[j][-1]

            elif action_category == "relative_init":
                g_i = np.eye(4)
                g_i[:3, 3] = init_eef_pos
                g_i[:3, :3] = init_eef_rotm

                g_a = np.eye(4)
                g_a[:3, 3] = actions[j][:3]
                g_a[:3, :3] = R.from_rotvec(actions[j][3:6]).as_matrix()

                g_rel = np.linalg.inv(g_i) @ g_a

                action = np.zeros(7)
                action[0:3] = g_rel[:3, 3]
                action[3:6] = R.from_matrix(g_rel[:3, :3]).as_rotvec()

                # gripper action
                action[-1] = actions[j][-1]

            elif action_category == "desired_frame":
                gd = np.eye(4)
                gd[:3, 3] = pd
                gd[:3, :3] = Rd

                g_a = np.eye(4)
                g_a[:3, 3] = actions[j][:3]
                g_a[:3, :3] = R.from_rotvec(actions[j][3:6]).as_matrix()

                g_rel = np.linalg.inv(gd) @ g_a

                action = np.zeros(7)
                action[0:3] = g_rel[:3, 3]
                action[3:6] = R.from_matrix(g_rel[:3, :3]).as_rotvec()

                # gripper action
                action[-1] = actions[j][-1]

            elif action_category == "desired_frame_correction":
                gd = np.eye(4)
                gd[:3, 3] = cubeA_pos_GT
                gd[:3, :3] = cubeA_rotm_GT

                g_a = np.eye(4)
                g_a[:3, 3] = actions[j][:3]
                g_a[:3, :3] = R.from_rotvec(actions[j][3:6]).as_matrix()

                g_rel = np.linalg.inv(gd) @ g_a

                action = np.zeros(7)
                action[0:3] = g_rel[:3, 3]
                action[3:6] = R.from_matrix(g_rel[:3, :3]).as_rotvec()

                # gripper action
                action[-1] = actions[j][-1]


            data_dict['/observations/state'].append(state)
            # data_dict['/observations/eef_vel'].append(obs[j]['eef_vel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(obs[j][f"{cam_name}_image"])

        pad_len = max_timesteps - ep_len

        if pad_len > 0:
            # Pad state and action
            last_state = data_dict['/observations/state'][-1]
            last_action = data_dict['/action'][-1]
            data_dict['/observations/state'] += [last_state.copy()] * pad_len
            data_dict['/action'] += [last_action.copy()] * pad_len

            # Pad images
            for cam_name in camera_names:
                last_image = data_dict[f'/observations/images/{cam_name}'][-1]
                data_dict[f'/observations/images/{cam_name}'] += [last_image] * pad_len



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
            eef_pos = obs.create_dataset('state', (max_timesteps, state.shape[0]))
            action = root.create_dataset('action', (max_timesteps, action.shape[0]))

            for name, array in data_dict.items():
                root[name][...] = array       
        print(f"Episode {episode_idx} saved to {dataset_path}.hdf5 in {time.time()-tic:.2f} seconds")
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ACT")
    parser.add_argument("--config_file", type=str, default="config/train/ACT_stack.yaml")

    args = parser.parse_args()

    main(vars(args))