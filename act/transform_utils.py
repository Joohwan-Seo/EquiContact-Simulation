import numpy as np
from scipy.spatial.transform import Rotation as R

def vee_map(mat):
    if mat.shape == (3, 3):
        # 3x3 matrix SO3 vee map
        return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
    elif mat.shape == (4, 4):
        # 4x4 matrix SE3 vee map
        # xi = [translation, rotation]
        return np.array([mat[3, 0], mat[3, 1], mat[3, 2], mat[2, 1], mat[0, 2], mat[1, 0]])

def hat_map(vec):
    if vec.shape == (3,):
        # 3D vector SO3 hat map
        return np.array([[0, -vec[2], vec[1]],
                           [vec[2], 0, -vec[0]],
                           [-vec[1], vec[0], 0]])

    elif vec.shape == (6,):
        # 6D vector SE3 hat map
        return np.array([[0, -vec[5], vec[4], vec[0]],
                           [vec[5], 0, -vec[3], vec[1]],
                           [-vec[4], vec[3], 0, vec[2]],
                           [0, 0, 0, 0]])
    
def SO3_log_map(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta == 0:
        xi = np.zeros(3,)
    else:
        xi = theta / (2 * np.sin(theta)) * vee_map(R - R.T)

    return xi

def SE3_log_map(g):
    p, R = g[:3,3], g[:3,:3]
    
    psi = SO3_log_map(R)
    psi_norm = np.linalg.norm(psi)
    psi_hat = hat_map(psi)

    if np.isclose(psi_norm, 0):
        A_inv = np.eye(3) - 0.5 * psi_hat + 1 / 12.0 * psi_hat @ psi_hat

    else:
        cot = 1 / np.tan(psi_norm / 2)
        alpha = (psi_norm /2) * cot

        A_inv = np.eye(3) - 0.5 * psi_hat + (1 - alpha)/(psi_norm**2) * psi_hat @ psi_hat

    v = A_inv @ p
    xi = np.zeros(6,)
    xi[:3] = v
    xi[3:] = psi

    return xi

def SE3_exp_map(xi):
    v, omega = xi[:3], xi[3:]

    omega_hat = hat_map(omega)

    xi_hat = np.zeros((4, 4))
    xi_hat[:3, :3] = omega_hat
    xi_hat[:3, 3] = v

    g = expm(xi_hat)

    return g

def quat_to_rot6d(quat):
    """Convert quaternion to 6D rotation representation.
    Args:
        quat (np.array): quaternion in wxyz format
    Returns:
        np.array: 6D rotation representation
    """
    r = R.from_quat(quat).as_matrix()

    return r[:3, :2].T.flatten()

def rotm_to_rot6d(rotm):
    return rotm[:3, :2].T.flatten()

def rotvec_to_rot6d(rotvec):
    r = R.from_rotvec(rotvec).as_matrix()

    return r[:3, :2].T.flatten()

def rot6d_to_quat(rot6d):
    """Convert 6D rotation representation to quaternion.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    print(f"x: {x}, y: {y}, z: {z}")
    quat = R.from_matrix(np.column_stack((x, y, z))).as_quat()
    
    return quat

def rot6d_to_rotm(rot6d):
    """Convert 6D rotation representation to rotation matrix.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)

    return np.column_stack((x, y, z))

def rot6d_to_rotvec(rot6d):
    """Convert 6D rotation representation to rotation vector.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    rotm = rot6d_to_rotm(rot6d)
    return R.from_matrix(rotm).as_rotvec()

def batch_pose_to_hom_matrices(poses):
    """
    poses: (N,6) array, each row = [tx,ty,tz,  ωx,ωy,ωz]
    returns: (N,4,4) array of homogeneous transforms
    """
    N = poses.shape[0]
    trans = poses[:, :3]             # (N,3)
    rotvecs = poses[:, 3:6]          # (N,3)

    # batch convert rotvecs → 3×3 rotation matrices
    R_mats = R.from_rotvec(rotvecs).as_matrix()  # (N,3,3)

    # build N copies of the 4×4 identity
    g_batch = np.tile(np.eye(4)[None, :, :], (N, 1, 1))  # (N,4,4)

    # insert translation and rotation
    g_batch[:, :3, 3] = trans
    g_batch[:, :3, :3] = R_mats

    return g_batch

def batch_rot6d_pose_to_hom_matrices(rot6d_poses):
    """
    rot6d_poses: (N,9) array, each row = [position, rot6d]
    returns: (N,4,4) array of homogeneous transforms
    """
    N = rot6d_poses.shape[0]
    pos = rot6d_poses[:, :3]             # (N,3)
    rot6d = rot6d_poses[:, 3:]          # (N,6)
    # Convert 6D rotation to rotation matrix

    R_mats = np.array([rot6d_to_rotm(rot6d_i) for rot6d_i in rot6d])  # (N,3,3)
    # build N copies of the 4×4 identity
    g_batch = np.tile(np.eye(4)[None, :, :], (N, 1, 1))  # (N,4,4)
    # insert translation and rotation
    g_batch[:, :3, 3] = pos
    g_batch[:, :3, :3] = R_mats

    return g_batch