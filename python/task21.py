import matplotlib.pyplot as plt
import numpy as np
from pose_estimation import *


XY01 = np.loadtxt("../data/platform_corners_metric.txt")
uv = np.loadtxt("../data/platform_corners_image.txt")
K = np.loadtxt("../data/K.txt")
heli_image = plt.imread('../data/video0000.jpg') #Image to plot estimations

if __name__ == "__main__":

    #Calulating xy from pixel coordinates
    uv1 = np.vstack((uv, np.ones(uv.shape[1])))
    xy = np.linalg.inv(K) @ uv1
    xy = xy[:2,:]/xy[2,:]

    XY = XY01[:2,:]
    XY1 = np.vstack((XY, np.ones(XY.shape[1])))

    #Estimating H and T = [R t]
    H = estimate_H(xy, XY)
    T1,T2 = decompose_H(H)
    T = T1 if np.all(T1@XY01 > 0) else T2

    #Calculating u_tilde for the two different transformations
    uv_H = project(K, H @ XY1)
    uv_Rt = project(K, T @ XY01)
    
    #Reprojection error
    error1 = uv_H - uv
    error2 = uv_Rt - uv

    print(np.round(error1, decimals=4))
    print(np.round(error2, decimals=4))

    plt.imshow(heli_image)
    plt.scatter(*uv_H, color='yellow', s=10, label='H')
    plt.scatter(*uv_Rt, color='red', s=10, label='[R t]')
    plt.legend()
    plt.show()
    plt.savefig("task21_scatter_plot")

