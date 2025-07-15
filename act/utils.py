import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            state = root['/observations/state'][start_ts]
            # qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        state_data = torch.from_numpy(state).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        state_data = (state_data - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]

        # noisify state and actions
        state_data = state_data + torch.randn_like(state_data) * 0.02
        action_data = action_data + torch.randn_like(action_data) * 0.02

        return image_data, state_data, action_data, is_pad
    
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class EpisodicDatasetFixed(Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats

        # Preload sim flag once
        first_id = self.episode_ids[0]
        first_path = os.path.join(self.dataset_dir, f'episode_{first_id}.hdf5')
        with h5py.File(first_path, 'r') as f:
            self.is_sim = bool(f.attrs['sim'])

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as f:
            # total timesteps
            L = f['/observations/state'].shape[0]
            # randomly pick start timestep
            start_ts = np.random.randint(L)

            # load state at start
            state = f['/observations/state'][start_ts][()]

            # load all camera images at start
            imgs = np.stack(
                [f[f'/observations/images/{cam}'][start_ts] for cam in self.camera_names],
                axis=0
            )  # shape: (K, H, W, C)

            # load pre-shaped action and pad mask
            action = f['/action'][start_ts][()]       # shape: (num_queries, action_dim)
            is_pad = f['/is_padded'][start_ts][()]    # shape: (num_queries,)

        # Convert to tensors with own storage
        img_t = torch.tensor(imgs, dtype=torch.uint8).permute(0,3,1,2).float().div(255).clone()
        state_t = torch.tensor(state, dtype=torch.float32).clone()
        action_t = torch.tensor(action, dtype=torch.float32).clone() # shape: (num_queries, action_dim)
        pad_t = torch.tensor(is_pad, dtype=torch.bool).clone()

        # Normalize
        state_t = (state_t - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
        action_t = (action_t - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # Add noise
        state_t = state_t + torch.randn_like(state_t) * 0.02
        action_t = action_t + torch.randn_like(action_t) * 0.02

        return img_t, state_t, action_t, pad_t


def get_norm_stats(dataset_dir, num_episodes):
    all_state_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            state = root['/observations/state'][()]
            # qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_state_data.append(torch.from_numpy(state))
        all_action_data.append(torch.from_numpy(action))

    all_state_data = torch.stack(all_state_data)
    all_action_data = torch.stack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    state_mean = all_state_data.mean(dim=[0, 1], keepdim=True)
    state_std = all_state_data.std(dim=[0, 1], keepdim=True)
    state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "state_mean": state_mean.numpy().squeeze(), "state_std": state_std.numpy().squeeze(),
             "example_qpos": state}

    return stats

def get_norm_stats_fixed(dataset_dir, num_episodes):
    all_state_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            state = root['/observations/state'][()]
            # qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_state_data.append(torch.from_numpy(state))
        all_action_data.append(torch.from_numpy(action))

    all_state_data = torch.stack(all_state_data)
    all_action_data = torch.stack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    state_mean = all_state_data.mean(dim=[0, 1], keepdim=True)
    state_std = all_state_data.std(dim=[0, 1], keepdim=True)
    state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "state_mean": state_mean.numpy().squeeze(), "state_std": state_std.numpy().squeeze(),
             "example_qpos": state}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats_fixed(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDatasetFixed(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDatasetFixed(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
