import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser import Quanser
from generate_quanser_summary import *


XY01 = np.loadtxt("../data/platform_corners_metric.txt")
uv = np.loadtxt("../data/platform_corners_image.txt")
K = np.loadtxt("../data/K.txt")
heli_image = plt.imread('../data/video0000.jpg') #Image to plot estimations

#Reused funcitons from assignment 4
def estimate_H(xy, XY):
    n = XY.shape[1]
    A = []
    for i in range(n):
        X,Y = XY[:,i]
        x,y = xy[:,i]
        A.append(np.array([X,Y,1, 0,0,0, -X*x, -Y*x, -x]))
        A.append(np.array([0,0,0, X,Y,1, -X*y, -Y*y, -y]))
    A = np.array(A)
    _,_,VT = np.linalg.svd(A)
    h = VT[8,:]
    H = np.reshape(h,(3,3))
    return H

def closest_rotation_matrix(Q):
    """
    Find closest (in the Frobenius norm sense) rotation matrix to 3x3 matrix Q
    """
    U,_,VT = np.linalg.svd(Q)
    R = U@VT
    return R

def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

def decompose_H(H):
    T1, T2 = np.eye(4), np.eye(4)
    k = np.linalg.norm(H[:,0])
    H /= k

    r1 = H[:,0]
    r2 = H[:,1]
    r3 = np.cross(r1, r2)
    t  = H[:,2]

    R1 = closest_rotation_matrix(np.column_stack((r1, r2, r3)))
    R2 = closest_rotation_matrix(np.column_stack((-r1, -r2, r3)))

    T1[:3,:3] = R1
    T2[:3,:3] = R2

    T1[:3,3] = t
    T2[:3,3] = -t

    return T1, T2

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
    u1 = project(K, H @ XY1)
    u2 = project(K, T @ XY01)
    
    #Reprojection error
    error1 = u1 - uv
    error2 = u2 - uv

    print(np.round(error1, decimals=4))
    print(np.round(error2, decimals=4))

    plt.imshow(heli_image)
    plt.scatter(*u1, linewidths=1, color='yellow', s=10, label='H')
    plt.scatter(*u2, color='red', label='[R t]', s=10)
    plt.legend()
    plt.show()
    plt.savefig("task21_scatter_plot")

