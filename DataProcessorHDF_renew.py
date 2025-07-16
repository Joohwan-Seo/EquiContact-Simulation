import os
from pathlib import Path
import cv2
import h5py
import argparse
import scipy
import json
from tqdm import tqdm 
import time 
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

import copy

from transform_utils import command_to_pose_data, command_to_pose_data_rot6d, vee_map, command_to_hom_matrix, hom_matrix_to_pose_data
from scipy.spatial.transform import Rotation as R


class DataProcessorHDFNew:
    def __init__(self, config): 
        self.task_config = config['task_config']
        self.train_config = config['train_config']

        #TODO(JS) NEED TO check
        self.dataset_dir = self.train_config['raw_dataset_dir'] # is length N list 
        self.save_dir = self.train_config['output_dataset_dir']
        self.reference_frame = self.train_config['reference_frame']

        self.reference_frame_tilted = self.train_config.get('reference_frame_tilted', self.reference_frame)

        self.state_category = self.task_config['state_category']
        self.state_dim = self.task_config['state_dim']
        self.action_category = self.task_config['action_category']
        self.action_dim = self.task_config['action_dim']
        self.camera_names = self.task_config['camera_names']
        self.logging_yaml = self.task_config['logging_yaml']
        self.num_queries = self.task_config['num_queries']

        self.ep_len = self.task_config['episode_len']

        self.noisy_reference = self.train_config['noisy_reference']
        self.noisy_reference_type = self.train_config['noisy_reference_type']

        self.crop_image = self.train_config['crop_image']
        self.crop_ratio = self.train_config['crop_ratio']
        self.mask_image = self.train_config['mask_image']
        self.mask_ratio = self.train_config['mask_ratio']

        self.fix_episode_length = self.train_config['fix_episode_length']

        self.task_name = self.task_config['task_name']

        if self.noisy_reference:
            self.noise_level_translation = self.train_config['noise_level_translation'] # in cm
            self.noise_level_rotation = self.train_config['noise_level_rotation'] # in degrees
        else:
            self.noise_level_translation = 0
            self.noise_level_rotation = 0

        # if rot6d exists, adjust pos_dim
        self.rot6d = self.task_config.get('rot6d', False)
        if self.rot6d:
            self.pose_dim = 9
        else:
            self.pose_dim = 6
        
    def process_data(self):
        # self.curr_dataset_dir = self.dataset_dir
        self.current_episode_idx = int(0)

        # Load the JSON file
        for current_dataset_dir in self.dataset_dir:
            self.curr_dataset_dir = current_dataset_dir
            json_file = os.path.join(current_dataset_dir, 'data.json')
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            # Process each episode
            for eps, eps_data in json_data.items():
                self.t0 = time.time()
                # print("current eps:", eps)
                # Skip ft_bias
                if eps != 'ft_bias':
                    try:
                        self.process_episode(eps, eps_data)
                        self.current_episode_idx += 1

                    except Exception as e:
                        print(f"Error in processing episode: {current_dataset_dir}, eps: {eps}")
                        print(e)

                        if e is KeyboardInterrupt:
                            print("KeyboardInterrupt: Exiting...")
                            break

                        continue
    
    def process_episode(self, eps, eps_data):
        # Indexing
        timesteps_list = list(eps_data.keys()) # Convert the timesteps to a list to index
        # print("Current_episode:", len(timesteps_list))

        # Get initial timestep and pose for reference
        init_timestep = timesteps_list[0] # Get the initial timestep
        # init_pose = eps_data[init_timestep]['robot_data']['observations'] # Get initial pose

        # Create a dictionary to store the data for each timestep
        data_dict = {
            '/observations/state': [],
            '/action': [],
            '/is_padded': [],
        }

        if self.fix_episode_length:
            ep_len = self.ep_len
        else:
            ep_len = len(timesteps_list) + self.num_queries

        # Get the all_action_data
        all_action_data = [eps_data[i]['proprio']['action'] for i in timesteps_list]
        all_obs_data = [eps_data[i]['proprio']['full_state'] for i in timesteps_list]

        # pad the last action data to the all_action_data for self.num_queries times
        all_action_data += [all_action_data[-1]] * (self.num_queries)
        all_padding_mask = np.zeros((ep_len,), dtype=bool)

        if self.task_name == "PIH_pick":
            # print("PIH_pick")
            all_padding_mask[len(timesteps_list) + 100:] = True
        else:
            all_padding_mask[len(timesteps_list):] = True

        
        if "tilted" in self.curr_dataset_dir:
            reference_frame = self.reference_frame_tilted
            # print(f"Processing tilted reference frame:{reference_frame}")
        else:
            reference_frame = self.reference_frame
            # print(f"Processing reference frame:{reference_frame}")


        # Extract data and store in data_dict
        for i in range(len(timesteps_list)):

            if i >= ep_len - self.num_queries:
                break

            # Access data
            timestep = timesteps_list[i]
            time_data = eps_data[timestep] 
            obs_data = all_obs_data[i]
            action_data = all_action_data[i]

            pd = np.array(reference_frame[0:3]) / 1000 # mm to m
            rotm_d = R.from_euler('xyz', reference_frame[3:], degrees=True).as_matrix()

            pd_GT = copy.deepcopy(pd)
            rotm_d_GT = copy.deepcopy(rotm_d)

            if self.noisy_reference:
                if self.noisy_reference_type == "fixed":
                    noise_pos = np.random.uniform(-1, 1, 3) * 0.01 * self.noise_level_translation
                    angle_noise = np.random.uniform(-1, 1, 3) * self.noise_level_rotation / 180 * np.pi # in radian
                    pd = pd_GT + noise_pos
                    rotm_d = rotm_d_GT @ R.from_euler('xyz', angle_noise, degrees=False).as_matrix()

            p = np.array(obs_data['ppos'][:3]) / 1000 # mm to m
            rotm = R.from_euler('xyz', obs_data['ppos'][3:], degrees=True).as_matrix()

            # Process states (proprioceptive info)
            states = []
            # print("State_category:", self.state_category)
            for state_name in self.state_category:
                if state_name == "world_pose":
                    if self.rot6d:
                        state = command_to_pose_data_rot6d(np.array(obs_data['ppos']))
                    else:
                        state = command_to_pose_data(np.array(obs_data['ppos'])) # to meter and rot6d()

                elif state_name == "world_vel":
                    state = np.array(obs_data['pvel'])
                
                elif state_name == "GCEV":
                    if self.noisy_reference:
                        if self.noisy_reference_type == "random":
                            noise_pos = np.random.uniform(-1, 1, 3) * 0.01 * self.noise_level_translation
                            angle_noise = np.random.uniform(-1, 1, 3) * self.noise_level_rotation / 180 * np.pi # in radian

                            pd = pd_GT + noise_pos
                            rotm_d = rotm_d_GT @ R.from_euler('xyz', angle_noise, degrees=False).as_matrix()

                        elif self.noisy_reference_type == "fixed":
                            pass

                        else:
                            raise ValueError(f"Unknown noisy reference type: {self.noisy_reference_type}")
                        
                    ep = rotm.T @ (p - pd)
                    eR = vee_map(rotm_d.T @ rotm - rotm.T @ rotm_d)

                    state = np.concatenate((ep, eR), axis=0)

                elif state_name == "FT":
                    state = np.array(obs_data['ft'])

                # print(f"state_name: {state_name}, state: {state}")
                states.append(state)

            processed_states = np.concatenate(states, axis=0)
            # print(f"processed_states shape: {processed_states.shape}")
            assert processed_states.shape == (self.state_dim,), f"State shape mismatch: {processed_states.shape} != {(self.state_dim,)}"


            actions = []
            for action_name in self.action_category:
                if action_name == "world_pose":
                    # action = np.zeros((self.num_queries, 6))
                    action = np.zeros((self.num_queries, self.pose_dim))
                    for j in range(self.num_queries):
                        world_pose_action = all_action_data[i + j]['cart_pose']
                        if self.rot6d:
                            action[j, :] = command_to_pose_data_rot6d(np.array(world_pose_action)) # to meter and rot6d()
                        else:
                            action[j, :] = command_to_pose_data(np.array(world_pose_action))

                elif action_name == "relative_pose":
                    #TODO To be implemented
                    g = np.eye(4)
                    g[:3,3] = p
                    g[:3,:3] = rotm
                    print(f"Processing pose: {p} and rotm: {rotm}")
                    action = np.zeros((self.num_queries, self.pose_dim))
                    for j in range(self.num_queries):
                        try:
                            world_pose_action = all_action_data[i + j]['cart_pose']
                        except:
                            print(len(timesteps_list))

                        # command to homogeneous matrix
                        g_action = command_to_hom_matrix(np.array(world_pose_action))

                        # homogeneous matrix = relative pose, relative pose to pose data (m, rotvec)
                        g_rel = np.linalg.inv(g) @ g_action
                        
                        action[j,:] = hom_matrix_to_pose_data(g_rel) # to meter and rotvec()

                elif action_name == "gains":
                    action = np.zeros((self.num_queries, 6))
                    for j in range(self.num_queries):
                        gains = all_action_data[i + j]['ad_gains']
                        action[j,: ] = np.log10(np.array(gains))

                elif action_name == "gripper":
                    action = np.zeros((self.num_queries, 1))
                    for j in range(self.num_queries):
                        gripper_action = all_action_data[i + j]['gripper_state']
                        action[j, :] = np.array(gripper_action)
                
                actions.append(action)

            current_padding_mask = all_padding_mask[i:i + self.num_queries]

            processed_action = np.concatenate(actions, axis=1)
            assert processed_action.shape == (self.num_queries, self.action_dim), f"Action shape mismatch: {processed_action.shape} != {(self.num_queries, self.action_dim)}"
            # print(f"processed_action shape: {processed_action.shape}")
            
            # Save data to data_dict
            data_dict['/observations/state'].append(processed_states)
            data_dict['/action'].append(processed_action)
            data_dict['/is_padded'].append(current_padding_mask)


            # Load the image data for each camera
            for camera_name in self.camera_names:
                key = f'/observations/images/{camera_name}'
                img_folder = os.path.join(self.curr_dataset_dir, 'images', camera_name)
                image = self.load_image(eps, timestep, camera_name, img_folder)

                if key not in data_dict:
                    data_dict[key] = []  

                data_dict[key].append(image)

        # Pad the sequences to the maximum length
        for key, seq in data_dict.items():
            while len(seq) < ep_len:
                seq.append(seq[-1])

        print(f"Episode {self.current_episode_idx} has {len(data_dict['/observations/state'])} timesteps.")
                
        # action_list = np.array(action_list)
        self.convert_to_hdf(data_dict)
        
    def convert_to_hdf(self, data_dict):
        max_timesteps = len(data_dict['/observations/state'])
        # print("max_timesteps:", max_timesteps)

        # Create the directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  

        # quit()

        # Count existing files to index the new dataset
        # idx = len([name for name in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, name))])
        save_path = os.path.join(self.save_dir, f'episode_{self.current_episode_idx}')
        print(f"Saving to {self.save_dir}")
        # Save the data to an HDF5 file, organizing datasets into groups
        with h5py.File(save_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in self.camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            state = obs.create_dataset('state', (max_timesteps, self.state_dim))
            action = root.create_dataset('action', (max_timesteps, self.num_queries, self.action_dim))
            is_padded = root.create_dataset('is_padded', (max_timesteps, self.num_queries), dtype=bool)
            for name, array in data_dict.items():    
                root[name][...] = array
        print(f'Processed episode {self.current_episode_idx} in {time.time() - self.t0:.1f} secs.')

    def load_image(self, episode, time_step, camera_name, image_folder, img_shape = (640, 480), debug=False):
        img_path = os.path.join(image_folder, f"{episode}_{camera_name}_{time_step}.png")
        if debug:
            print(f"Loading image: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != img_shape:
            image = cv2.resize(image, img_shape)

        if self.crop_image:
            image = self.center_crop(image, self.crop_ratio)
        
        if self.mask_image:
            image = self.boundary_mask(image, self.mask_ratio)
        
        return image
    
    def center_crop(self, image, ratio = 0.8):
        """
        center crop and resize to the original one
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * ratio), int(w * ratio)
        start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
        cropped_image = image[start_h:start_h + new_h, start_w:start_w + new_w]
        return cv2.resize(cropped_image, (w, h))  # Resize to original size

    def boundary_mask(self, image, mask_width_frac=0.5):
        """
        Apply a soft cosine “vignette” mask around the image borders.
        mask_width_frac: fraction of width/height to taper out (0–0.5)
        """
        h, w = image.shape[:2]
        # mask widths in pixels
        mw, mh = int(w * mask_width_frac), int(h * mask_width_frac)

        # build 1D cosine taper windows
        x = np.linspace(-np.pi, np.pi, w)
        y = np.linspace(-np.pi, np.pi, h)
        wx = 0.5 * (1 + np.cos(np.clip(x / (mask_width_frac * np.pi), -np.pi, np.pi)))
        wy = 0.5 * (1 + np.cos(np.clip(y / (mask_width_frac * np.pi), -np.pi, np.pi)))

        # outer product to get 2D mask
        vignette = np.outer(wy, wx)
        vignette = vignette[:, :, np.newaxis]  # shape (h, w, 1)

        # apply mask
        masked = (image.astype(np.float32) * vignette).astype(image.dtype)
        return masked
    
    def show_statistics(self):
        # return the number of episodes
        print(f"Datasets directory: {self.dataset_dir}")

        length_episodes = []
        total_episode = 0

        for dataset in self.dataset_dir:
            if "tilted" in dataset:
                print("Processing tilted reference frame")
            json_file = os.path.join(dataset, 'data.json')
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                for eps, eps_data in json_data.items():
                    length_episodes.append(len(eps_data))
                    total_episode += 1
                    print(f"Episode: {eps}")
                    print(f"Number of timesteps: {len(eps_data)}")      

        print(f"Number of episodes: {len(length_episodes)}")
        print(f"Average number of timesteps: {np.mean(length_episodes)}")
        print(f"Max number of timesteps: {np.max(length_episodes)}")
               
def main(args):
    dp = DataProcessorHDFNew(config)
    # for 20 series
    # MAX number of timesteps: 449
    # Number of episodes: 137

    # dp.show_statistics()

    # for pattern series
    # MAX number of timesteps: 636
    # Number of episodes: 83

    # for pattern + platform series
    # max number of timesteps: 762
    # number of episodes: 85

    # for pick series
    # MAX number of timesteps: 922
    # Number of episodes: 48

    dp.process_data()

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config" / "ACT_realrobot_pattern_platform_06_distractor_tilted.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

    print(config['train_config']['output_dataset_dir'])
    print(config['task_config']['task_name'])
    main(config)

    # config_path = Path(__file__).parent / "config" / "ACT_realrobot_02.yaml"
    # config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    # main(config)