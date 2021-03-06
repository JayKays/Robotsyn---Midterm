import matplotlib.pyplot as plt
import numpy as np
from pose_estimation import *
from methods import levenberg_marquardt


XY01 = np.loadtxt("../data/platform_corners_metric.txt")
uv = np.loadtxt("../data/platform_corners_image.txt")
K = np.loadtxt("../data/K.txt")
heli_image = plt.imread('../data/video0000.jpg') #Image to plot estimations


def pose(p, R0):
    ''' Calculates the pose from parametrization'''
    T = np.eye(4)

    t = p[3:]

    s, c = np.sin, np.cos
    Rx = lambda a: np.array([[1, 0, 0], [0, c(a), s(a)],[0, -s(a), c(a)]])
    Ry = lambda a: np.array([[c(a), 0, -s(a)], [0, 1, 0],[s(a), 0, c(a)]])
    Rz = lambda a: np.array([[c(a), s(a), 0], [-s(a), c(a), 0],[0, 0, 1]])

    R = Rx(p[0]) @ Ry(p[1]) @ Rz(p[2]) @ R0


    T[:3,:3] = R
    T[:3,3] = t

    return T

def residual(p, R0):
    '''Calculate projection residuals from parametrization'''
    T = pose(p,R0)
    uv_hat = project(K, T @ XY01)
    r = uv_hat - uv
    return np.ravel(r)

if __name__ == "__main__":

    #XY coordinates with/without added 1
    XY = XY01[:2,:]
    XY1 = np.vstack((XY, np.ones(XY.shape[1])))

    #Calulating xy from pixel coordinates
    uv1 = np.vstack((uv, np.ones(uv.shape[1])))
    xy = np.linalg.inv(K) @ uv1
    xy = xy[:2,:]/xy[2,:]

    #Estimating H and T = [R t]
    H = estimate_H(xy, XY)
    T1,T2 = decompose_H(H)
    T = T1 if np.all((T1@XY01)[2,:] >= 0) else T2

    #Task 2.1 points for comparison
    uv_H = project(K, H @ XY1)
    uv_Rt = project(K, T @ XY01)

    #Initial pose and parametrization
    R0 = T[:3,:3]
    t0 = T[:3,3]
    #t0 = [1.,1.,1.]
    p0 = np.hstack(([0,0,0], t0))
    print(f"Initial p: {p0}")

    #LM estimation
    p = levenberg_marquardt(lambda p: residual(p,R0), p0, tol = 1e-12)

    #Calculate ouput from LM estimation
    T_LM = pose(p,R0)
    uv_LM = project(K, T_LM@XY01)
    

    # print(np.round(uv_Rt - uv, decimals = 3))
    # print(np.round(uv_LM - uv, decimals = 3))

    print(np.linalg.norm(uv_Rt - uv))
    print(np.linalg.norm(uv_LM - uv))

    #Generate plot
    plt.imshow(heli_image)
    plt.scatter(*uv_H, linewidths=1, color='red', s=10, label='H')
    plt.scatter(*uv_Rt, color='blue',s=10, label='[R t]')
    plt.scatter(*uv_LM, linewidths=1, color='green', s=10, label='LM')
    # plt.scatter(*uv, color='red', label='Detection', s=10)
    plt.legend()
    plt.axis([200, 500, 600, 400])
    plt.savefig("task22_scatter_plot")
    plt.show()

