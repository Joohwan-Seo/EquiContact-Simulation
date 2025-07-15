import numpy
from scipy.spatial.transform import Rotation as R

def vee_map(mat):
    if mat.shape == (3, 3):
        # 3x3 matrix SO3 vee map
        return numpy.array([mat[2, 1], mat[0, 2], mat[1, 0]])
    elif mat.shape == (4, 4):
        # 4x4 matrix SE3 vee map
        # xi = [translation, rotation]
        return numpy.array([mat[3, 0], mat[3, 1], mat[3, 2], mat[2, 1], mat[0, 2], mat[1, 0]])

def hat_map(vec):
    if vec.shape == (3,):
        # 3D vector SO3 hat map
        return numpy.array([[0, -vec[2], vec[1]],
                           [vec[2], 0, -vec[0]],
                           [-vec[1], vec[0], 0]])

    elif vec.shape == (6,):
        # 6D vector SE3 hat map
        return numpy.array([[0, -vec[5], vec[4], vec[0]],
                           [vec[5], 0, -vec[3], vec[1]],
                           [-vec[4], vec[3], 0, vec[2]],
                           [0, 0, 0, 0]])