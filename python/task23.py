import matplotlib.pyplot as plt
import numpy as np
from pose_estimation import *
from methods import levenberg_marquardt
from part2 import pose

XY01 = np.loadtxt("../data/platform_corners_metric.txt")[:,:-1]
uv = np.loadtxt("../data/platform_corners_image.txt")[:,:-1]
K = np.loadtxt("../data/K.txt")
heli_image = plt.imread('../data/video0000.jpg') #Image to plot estimations

def residual(p, R0):
    '''Calculate residuals from parametrization'''
    T = pose(p,R0)
    uv_hat = project(K, T @ XY01[:,:3])
    
    r = uv_hat - uv
    return np.ravel(r)

if __name__ == "__main__":

    #XY coordinates 
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

    #Initial pose and parametrization 1st pose
    R01 = T.copy()[:3,:3]
    t01 = T.copy()[:3,3]

    p01 = np.hstack(([0,0,0], t01)) 
    p1 = levenberg_marquardt(lambda p: residual(p, R01), p01, tol = 1e-12)
    pose1 = pose(p1,R01)

    print("pose 1:")
    print(pose1)

    #Initial pose and parametrization 2nd pose
    R02 = T.copy()[:3,:3].T
    t02 = T.copy()[:3,3]

    p02 = np.hstack(([0,0,0], t02)) 
    p2 = levenberg_marquardt(lambda p: residual(p,R02), p02, tol = 1e-12)
    pose2 = pose(p2,R02)

    print("pose 2:")
    print(np.round(pose2, decimals = 4))

    #Pixel coordinates of the fromt the two poses
    uv1 = project(K, pose1@XY01[:,:])
    uv2 = project(K, pose2@XY01[:,:])

    print("Rep. error pose 1 :", np.round(np.linalg.norm(uv1 - uv, axis = 0), decimals = 10))
    print("Rep. error pose 2: ", np.round(np.linalg.norm(uv2 - uv, axis = 0), decimals = 10))
    

    #pose from task 2.2
    T_task22 = np.array([[ 0.89361792, -0.44858958,  0.01464272, -0.2582405 ],\
                         [-0.09159264, -0.21420173, -0.97248569,  0.11634566],\
                         [ 0.43938344,  0.86768947, -0.23250201,  0.79018261],\
                         [0., 0., 0., 1.]])

    #reloading files with all 4 points to plot
    uv = np.loadtxt("../data/platform_corners_image.txt")
    XY01 = np.loadtxt("../data/platform_corners_metric.txt")
    uv1 = project(K, pose1@XY01[:,:])
    uv2 = project(K, pose2@XY01[:,:])

    plt.imshow(heli_image)
    plt.title("Plotting all 4 points with 3 point optimized transformation")
    plt.scatter(*uv, linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
    plt.scatter(*uv1, linewidths=1, color='lime', s=40, label='Pose 1')
    plt.scatter(*uv2, color='blue',s=10, label='Pose 2')
    plt.legend()
    plt.axis([200, 500, 600, 400])
    plt.show()
