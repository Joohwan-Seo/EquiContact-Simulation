import pickle

import matplotlib.pyplot as plt

import os
import numpy as np

from scipy.spatial.transform import Rotation as R

# import opencv

def main():
    # load the dataset #1

    dir = "data_collection/datasets/stack/s_GCEV_a_relative_fixed_random_noise"

    # with open(dir, 'rb') as f:
    #     rollouts = pickle.load(f)


    # obs = rollouts['observations']
    # action = rollouts['actions']

    # # print(obs)
    # eef_pos = obs[0]['robot0_eef_pos']
    # eef_quat = obs[0]['robot0_eef_quat_site']
    # eef_rotvec = R.from_quat(eef_quat).as_rotvec()

    # state = np.concatenate((eef_pos, eef_rotvec))
    # print(state)
    # print(action[0])

    # plt.imshow(obs[0]['robot0_eye_in_hand_image'])
    # plt.show()

    eef_pos_list = []
    eef_rotvec_list = []
    action_list = []
    prev_eef_pos = np.zeros(3,)
    delta_eef_list = []
    for i in range(1):
        name = dir + f"/demo_{i+1}.pkl"
        with open(name, 'rb') as f:
            rollouts = pickle.load(f)
        
        obs = rollouts['observations']
        print(obs.keys())
        actions = rollouts['actions']

        action_list.append(actions)

        for j in range(len(obs)):
            eef_pos = obs[j]['robot0_eef_pos']
            eef_delta = eef_pos - prev_eef_pos
            print(eef_delta)
            prev_eef_pos = eef_pos

            delta_eef_list.append(eef_delta)

    # get the mean and std of the actions
    action_list = np.array(action_list)
    action_mean = np.mean(action_list, axis=0)
    action_std = np.std(action_list, axis=0)

    delta_eef_list = delta_eef_list[1:]
    eef_delta = np.array(delta_eef_list)

    print("eef_delta_mean", eef_delta.mean(axis=0))
    print("eef_delta_std", eef_delta.std(axis=0))
    print("eef_delta_sample:", eef_delta[:5,:])

    print("action_mean", action_mean)
    print("action_std", action_std)




        


if __name__ == "__main__":

    main()