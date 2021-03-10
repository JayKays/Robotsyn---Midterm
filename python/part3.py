import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from methods import jacobian, levenberg_marquardt
from common import *
from quanser import Quanser


detections = np.loadtxt('../data/detections.txt')
heli_points = np.loadtxt('../data/heli_points.txt').T
K = np.loadtxt('../data/K.txt')
platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')

def marker_poses(statics, angles):
    '''Calculates arm and rotors to camera poses for given variables'''
    #statics = 5 lengths + heli points
    base_to_platform = translate(statics[0]/2, statics[0]/2, 0.0)@rotate_z(angles[0])
    hinge_to_base    = translate(0.00, 0.00,  statics[1])@rotate_y(angles[1])
    arm_to_hinge     = translate(0.00, 0.00, -statics[2])
    rotors_to_arm    = translate(statics[3], 0.00, -statics[4])@rotate_x(angles[2])

    base_to_camera   = platform_to_camera @ base_to_platform
    hinge_to_camera  = base_to_camera @ hinge_to_base
    arm_to_camera    = hinge_to_camera @ arm_to_hinge
    rotors_to_camera = arm_to_camera @ rotors_to_arm

    return rotors_to_camera, arm_to_camera

def generalized_poses(statics, angles):
    '''Calculates fully parametrized helicopter poses'''
    #TODO: Gjøre dette, ingen anelse hva de mener atm.

    #Statics = (2 lengths + 2 angles)*3 + 1 length + helipoints

    stat_length = statics[:8]
    stat_angle = statics[8:14]

    base_to_platform = translate(stat_length[0]/2, stat_length[1]/2, 0.0) @ \
        rotate_x(stat_angle[0]) @ rotate_y(stat_angle[1]) @ rotate_z(angles[0])

    hinge_to_base    = translate(stat_length[2], 0.00,  stat_length[3]) @\
        rotate_x(stat_angle[2]) @ rotate_z(stat_angle[3]) @ rotate_y(angles[1])

    arm_to_hinge     = translate(0.00, 0.00, -stat_length[4])

    rotors_to_arm    = translate(stat_length[5], stat_length[6], -stat_length[7])@\
        rotate_y(stat_angle[4]) @ rotate_z(stat_angle[5]) @ rotate_x(angles[2])

    base_to_camera   = platform_to_camera @ base_to_platform
    hinge_to_camera  = base_to_camera @ hinge_to_base
    arm_to_camera    = hinge_to_camera @ arm_to_hinge
    rotors_to_camera = arm_to_camera @ rotors_to_arm

    return rotors_to_camera, arm_to_camera

def image_residuals(statics, angles, uv, weights, generalize):
    '''Calculates the residuals of a given image 
    for static parameters and dynamic angles'''
    
    marker_points = np.vstack((np.reshape(statics[-21:], (3,7)), np.ones(7)))

    T_rc, T_ac = generalized_poses(statics, angles) if generalize else marker_poses(statics, angles)

    # Compute the predicted image location of the markers with given angles and lengths
    p1 = T_ac @ marker_points[:,:3]
    p2 = T_rc @ marker_points[:,3:]
    uv_hat = project(K, np.hstack([p1, p2]))

    r = (uv_hat - uv)*weights
    
    return np.ravel(r)

def residuals(p, l, m, generalize):
    '''Calculates the total residuals for all images'''
    r = np.zeros(2*7*l)
    statics = p[:m]
    dynamics = p[m:]

    for i in range(l):
        angles = dynamics[3*i: 3*(i+1)]
        weights = detections[i, ::3]
        uv = np.vstack((detections[i, 1::3], detections[i, 2::3]))
        r[2*7*i:2*7*(i+1)] = image_residuals(statics, angles,uv, weights, generalize)

    return r

def jac_blocks(p, eps, l, m, generalize):
    '''Calculates the 2nl x m block and the
       2n x 3 blocks of the jacobian matrix'''

    n = 7
    statics = p[:m]
    dynamics = p[m:]

    static_jac = np.zeros((2*n*l, m))
    dyn_jacs = np.zeros((2*n, 3, l))
    # weights = detections[:, ::3]

    for i in range(l):
        angles = dynamics[3*i: 3*(i+1)]
        uv = np.vstack((detections[i, 1::3], detections[i, 2::3]))
        weights = detections[i, ::3]

        #2n x m and 2n x 3 blocks of the jacobian matrix
        #Static Jacobian block
        im_res1 = lambda x: image_residuals(x, angles, uv, weights, generalize)
        static_jac[2*n*i:2*n*(i+1) , :] = jacobian(im_res1, statics, eps)
        
        #Dynamic Jacobian block
        im_res2 = lambda x: image_residuals(statics, x, uv, weights, generalize)
        dyn_jacs[:,:,i] = jacobian(im_res2, angles, eps)
        

    return static_jac, dyn_jacs

def hessian_blocks(static_jac, dyn_jacs, mu):
    '''Calculates the blocks in the approximate Hessian from 
    the jacobian blocks, with added damping mu * I'''

    l = dyn_jacs.shape[2]
    m = static_jac.shape[1]

    A11 = static_jac.T @ static_jac + mu*np.eye(m)
    A12 = static_jac.T @ block_diag(*(dyn_jacs[:,:,i].copy() for i in range(l)))

    A22 = np.zeros((3,3,l))
    for i in range(l):
        A22[:, :, i] = dyn_jacs[:,:, i].T @ dyn_jacs[:,:,i] + np.eye(3)*mu

    return A11, A12, A22

def schurs_sol(stat, dyn, A11,A12,A22, r):

    '''Calculates the solution to the normal equation
        using schurs complement, assuming linear system of the form
        A *x + B*y = a
        Bt*x + D*y = b
    '''
    l = int(A12.shape[1]/3)

    #Converting sparse blocks to block diagonal matrices
    
    Dyn = block_diag(*(dyn[:,:,i] for i in range(l)))
    D_inv = block_diag(*(np.linalg.inv(A22[:,:,i]) for i in range(l)))
    D = block_diag(*(A22[:,:,i] for i in range(l)))

    #Init a,b where [a b]^T = -J^t * r
    
    a =  -1 * stat.T @ r
    b =  -1 * Dyn.T @ r

    schurs = A11 - A12 @ D_inv @ A12.T
    delta_stat = np.linalg.solve(schurs, a - A12 @ D_inv @ b)
    delta_dyn = np.linalg.solve(D, b - A12.T @ delta_stat)

    #delta = [x y]^T
    return np.hstack((delta_stat, delta_dyn))


def optimize(residualsfun, p0, generalize = False, max_iterations=100, tol = 1e-6, finite_difference_epsilon=1e-5):
    
    E = lambda r: np.sum(r**2)

    m = 26 if not generalize else 35
    l = (p0.shape[0] - m)//3

    p = p0.copy()

    static_jac, dyn_jacs = jac_blocks(p, finite_difference_epsilon, l, m, generalize)
    A11, A12, A22 = hessian_blocks(static_jac, dyn_jacs, mu = 0)
    mu = 1e-3 * np.maximum(np.amax(A11.diagonal()), np.amax([np.amax(A22[:,:,i].diagonal()) for i in range(l)]))
    
    for _ in range(max_iterations):

        r = residualsfun(p)
        # print("Res shape: ", r.shape)

        static_jac, dyn_jacs = jac_blocks(p, finite_difference_epsilon, l, m, generalize)
        A11, A12, A22 = hessian_blocks(static_jac, dyn_jacs, mu)

        delta = schurs_sol(static_jac, dyn_jacs, A11, A12, A22, r)

        #Updates mu until delta is accepted
        while E(r) < E(residualsfun(p + delta)):
            # print(f"MU doubled, step {_}")
            mu *= 2
            static_jac, dyn_jacs = jac_blocks(p, finite_difference_epsilon, l, m, generalize)
            A11, A12, A22 = hessian_blocks(static_jac, dyn_jacs, mu)

            delta = schurs_sol(static_jac, dyn_jacs, A11,A12, A22, r)

        #Perform step
        print(f"Step {_}:\t E(p) =  {np.round(E(r), decimals = 6)}\t |delta| = {np.round(np.linalg.norm(delta), decimals = 6)}")



        p += delta
        mu /= 3

        if (np.linalg.norm(delta) < tol): break
        if (E(r) - E(residualsfun(p)) < tol):
            print(E(residualsfun(p)))
            break

        #Stopping criteria


    return p

def get_init_traj(l):
    '''Copy of code from task 1 to calculate initial trajectory over l images'''
    quanser = Quanser()
    p = np.array([11.6, 28.9, 0.0])*np.pi/180
    trajectory = np.zeros((l, 3))
    for image_number in range(l):
        weights = detections[image_number, ::3]
        uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))

        residualsfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])
        p = levenberg_marquardt(residualsfun, p)

        trajectory[image_number, :] = p

    return np.ravel(trajectory)


def plot_heli_points(p, image_number, m, general = False, name = "", col = 'red'):
    '''Generates plot of marker points from p over a given image number'''

    statics = p[:m]
    angles = p[m + image_number*3: m + (image_number+1)*3]
    heli_image = plt.imread('../data/video%04d.jpg' % image_number)
    
    T_rc, T_ac = generalized_poses(statics, angles) if general else marker_poses(statics, angles)
    marker_points = np.vstack((np.reshape(p[m-21: m], (3,7)), np.ones(7)))
    # print(np.round(marker_points, decimals = 5))
    p1 = T_ac @ marker_points[:,:3]
    p2 = T_rc @ marker_points[:,3:]

    uv = project(K, np.hstack((p1,p2)))

    plt.imshow(heli_image)
    plt.scatter(*uv, linewidths=1, color = col, s=10, label=name)

def optimize_model(l, general = False, plot_points = False, image = 1):
    '''Runs the optimization and returns optimized static parameters'''

    #initialize p0
    if general:
        stat_lengths = np.array([0.1145, 0.1145, 0.0, 0.325,\
             0.050, 0.65, 0.0, 0.030])
        stat_angles = np.zeros(6)
        markers = np.ravel(heli_points[:3,:])
        static = np.hstack((stat_lengths, stat_angles, markers))

    else: 
        lengths = np.array([0.1145, 0.325, 0.050, 0.65, 0.030])
        markers = np.ravel(heli_points[:3,:])
        static = np.hstack((lengths, markers))
        
    dynamic = get_init_traj(l)

    p0 = np.hstack((static, dynamic))
    m = p0.shape[0] - dynamic.shape[0]

    res = lambda p: residuals(p, l, m, general)
    print("Init complete, Optimizing model")
    p = optimize(res, p0, generalize = general)

    params = p[:m-21]
    points = np.vstack((np.reshape(p[m-21: m], (3,7)), np.ones(7)))

    if plot_points:
        plot_heli_points(p0, image, m, general, "p0", 'yellow')
        plot_heli_points(p, image, m, general, "p", 'red')
        plt.legend()
        plt.show()

    return params, points


if __name__ == "__main__":
    generalize = True
    l = detections.shape[0]
    visualize_image = 1

    params, points = optimize_model(l, general = generalize, plot_points= True, image = visualize_image)

    print(params)
    print(points)

