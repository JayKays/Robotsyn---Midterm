import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from methods import jacobian, levenberg_marquardt
from common import *  # noqa  # pylint: disable=unused-import
from quanser import Quanser
del globals()["draw_frame"]

detections = np.loadtxt('../data/detections.txt')
heli_points = np.loadtxt('../data/heli_points.txt').T
K = np.loadtxt('../data/K.txt')
platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')

def marker_poses(statics, angles):
    '''Calculates arm and rotors to camera poses for given variables'''

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

    base_to_platform = translate(statics[0]/2, statics[0]/2, 0.0)@rotate_z(angles[0])
    hinge_to_base    = translate(0.00, 0.00,  statics[1])@rotate_y(angles[1])
    arm_to_hinge     = translate(0.00, 0.00, -statics[2])
    rotors_to_arm    = translate(statics[3], 0.00, -statics[4])@rotate_x(angles[2])

    base_to_camera   = platform_to_camera @ base_to_platform
    hinge_to_camera  = base_to_camera @ hinge_to_base
    arm_to_camera    = hinge_to_camera @ arm_to_hinge
    rotors_to_camera = arm_to_camera @ rotors_to_arm

    return rotors_to_camera, arm_to_camera

def image_residuals(statics, angles, uv, weights):
    '''Calculates the residuals for a given image'''

    lengths = statics[:5]
    marker_points = np.vstack((np.reshape(statics[5:], (3,7)), np.ones(7)))
    T_rc, T_ac = marker_poses(lengths, angles)

    # Compute the predicted image location of the markers with given angles and lengths
    p1 = T_ac @ marker_points[:,:3]
    p2 = T_rc @ marker_points[:,3:]
    uv_hat = project(K, np.hstack([p1, p2]))

    r = (uv_hat - uv)*weights

    return np.ravel(r)

def residuals(p, l, m):
    '''Calculates the total residuals for all images'''
    r = np.zeros(2*7*l)
    # weights = detections[:, ::3]
    statics = p[:m]
    dynamics = p[m:]

    for i in range(l):
        angles = dynamics[3*i: 3*(i+1)]
        weights = detections[i, ::3]
        uv = np.vstack((detections[i, 1::3], detections[i, 2::3]))
        r[2*7*i:2*7*(i+1)] = image_residuals(statics, angles,uv, weights)

    return r

def jac_blocks(p, eps, l, m):
    '''Calculates the 2nl x m block and the
       2n x 3 blocks of the jacobian matrix'''

    n = 7
    test = p.copy()
    statics = test[:m]
    dynamics = test[m:]

    static_jac = np.zeros((2*n*l, m))
    dyn_jacs = np.zeros((2*n, 3, l))
    # weights = detections[:, ::3]

    for i in range(l):
        angles = dynamics[3*i: 3*(i+1)]
        uv = np.vstack((detections[i, 1::3], detections[i, 2::3]))
        weights = detections[i, ::3]

        #2n x m and 2n x 3 blocks of the jacobian matrix
        #Static Jacobian block
        im_res1 = lambda x: image_residuals(x, angles, uv, weights)
        static_jac[2*n*i:2*n*(i+1) , :] = jacobian(im_res1, statics, eps)

        #Dynamic Jacobian block
        im_res2 = lambda x: image_residuals(statics, x, uv, weights)
        dyn_jacs[:,:,i] = jacobian(im_res2, angles, eps)

    return static_jac, dyn_jacs

def hessian_blocks(static_jac, dyn_jacs, mu):
    '''Calculates the blocks in the approximate Hessian from 
    the jacobian blocks, with added damping mu * I'''

    l = dyn_jacs.shape[2]
    m = static_jac.shape[1]

    A11 = static_jac.T @ static_jac + mu*np.eye(m)
    A12 = np.zeros((m,3*l))
    A22 = np.zeros((3,3,l))

    for i in range(l):
        A12[:, i:(i+3)] = static_jac[14*i: 14*(i+1),:].T @ dyn_jacs[:,:,i]
        A22[:, :, i] = dyn_jacs[:,:, i].T @ dyn_jacs[:,:,i] + np.eye(3)*mu

    return A11, A12, A22


def schurs_sol(stat, dyn, A11,A12,A22, r):

    '''Calculates the solution to the normal equation
        using schurs complement, assuming linear system of the form
        A *x + B*y = a
        Bt*x + D*y = b
    '''
    l = int(A12.shape[1]/3)
    m = A11.shape[0]
    # n = int(dyn.shape[0]/2) 

    #init matrices in schurs complement and inverse block digaonal
    BDBt = np.zeros((m,m))
    BD_inv = np.zeros((m, 3*l))
    D_inv = np.zeros(A22.shape)
    # D_inv = block_diag(*(np.linalg.inv(A22[:,:,i]) for i in range(l)))
    #Init a,b where [a b]^T = -J^t * r
    r_neg = (-1) * r.copy()
    a = stat.T @ r_neg
    b = np.zeros(3*l)
    for i in range(l):
        b[3*i: 3*(i+1)] = dyn[:,:,i].T @ r_neg[14*i:14*(i+1)]
        B_block = A12[:,3*i:3*(i+1)]
        d_inv = np.linalg.inv(A22[:,:,i])
        D_inv[:,:,i] = d_inv

        BDBt += B_block @ d_inv @ B_block.T
        BD_inv[:,3*i:3*(i+1)] = B_block @ d_inv

    #Solves for x = static opt. variables with schurs complement
    delta_stat = np.linalg.solve(A11 - BDBt, a - BD_inv @ b)

    #Use delta_stat to find solution for y = dynamic opt. variables
    #y = D^-1(b - B^T * x)
    delta_dyn = np.zeros(3*l)
    for i in range(l):
        bCx = b[3*i: 3*(i+1)] - A12[:, 3*i: 3*(i+1)].T @ delta_stat
        delta_dyn[3*i: 3*(i+1)] = D_inv[:,:,i] @ bCx

    #delta = [x y]^T
    return np.hstack((delta_stat, delta_dyn))


def optimize(residualsfun, p0, max_iterations=100, tol = 1e-6, finite_difference_epsilon=1e-5):

    E = lambda r: np.sum(r**2)
    m = 5 + 3*7
    l = (p0.shape[0] - m)//3

    p = p0.copy()

    static_jac, dyn_jacs = jac_blocks(p, finite_difference_epsilon, l, m)
    A11, A12, A22 = hessian_blocks(static_jac, dyn_jacs, mu = 0)
    mu = 1e-3 * np.maximum(np.amax(A11.diagonal()), np.amax([np.amax(A22[:,:,i].diagonal()) for i in range(l)]))

    for _ in range(max_iterations):

        r = residualsfun(p)
        # print("Res shape: ", r.shape)

        static_jac, dyn_jacs = jac_blocks(p, finite_difference_epsilon, l, m)
        A11, A12, A22 = hessian_blocks(static_jac, dyn_jacs, mu)

        delta = schurs_sol(static_jac, dyn_jacs, A11,A12, A22, r)

        #Updates mu until delta is accepted
        while E(r) < E(residualsfun(p + delta)):
            # print(f"MU doubled, step {_}")
            mu *= 2
            static_jac, dyn_jacs = jac_blocks(p, finite_difference_epsilon, l, m)
            A11, A12, A22 = hessian_blocks(static_jac, dyn_jacs, mu)

            delta = schurs_sol(static_jac, dyn_jacs, A11,A12, A22, r)

        #Perform step
        print(f"Step {_}:\t E(p) =  {np.round(E(r), decimals = 6)}\t |delta| = {np.round(np.linalg.norm(delta), decimals = 6)}")

        p += delta
        mu /= 3

        if np.linalg.norm(delta) < tol : break

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


def plot_heli_points(p, image_number, m, name = "", col = 'red'):
    '''Generates plot of marker points from p over a given image number'''

    statics = p[:m]
    angles = p[m + image_number*3: m + (image_number+1)*3]
    heli_image = plt.imread('../data/video%04d.jpg' % image_number)

    T_rc, T_ac = marker_poses(statics, angles)
    marker_points = np.vstack((np.reshape(p[5: m], (3,7)), np.ones(7)))
    # print(np.round(marker_points, decimals = 5))
    p1 = T_ac @ marker_points[:,:3]
    p2 = T_rc @ marker_points[:,3:]

    uv = project(K, np.hstack((p1,p2)))

    plt.imshow(heli_image)
    plt.scatter(*uv, linewidths=1, color = col, s=10, label=name)

def fetch_optimized_params(l):
    '''Function to pass optimize params to task1 script'''

    generalize = False
    # l = detections.shape[0]

    #initial p
    if generalize:
        lengths = np.array([0.1145, 0.325, 0.050, 0.65, 0.030])
        markers = np.ravel(heli_points[:3,:])
        angles = get_init_traj(l)
    else: 
        lengths = np.array([0.1145, 0.325, 0.050, 0.65, 0.030])
        markers = np.ravel(heli_points[:3,:])
        angles = get_init_traj(l)

    m = lengths.shape[0] + markers.shape[0]
    
    p0 = np.hstack((lengths, markers, angles))

    res = lambda p: residuals(p, l, m)
    print("Init complete, running batch optimization")
    p = optimize(res, p0)

    lengths = p[:5]
    points = np.vstack((np.reshape(p[5: m], (3,7)), np.ones(7)))

    return lengths, points


if __name__ == "__main__":
    generalize = False
    l = detections.shape[0]
    visualize_image = 1
    # l = 100

    #initial p
    if generalize:
        lengths = np.array([0.1145, 0.325, 0.050, 0.65, 0.030])
        markers = np.ravel(heli_points[:3,:])
        angles = get_init_traj(l)
    else: 
        lengths = np.array([0.1145, 0.325, 0.050, 0.65, 0.030])
        markers = np.ravel(heli_points[:3,:])
        angles = get_init_traj(l)

    p0 = np.hstack((lengths, markers, angles))

    m = lengths.shape[0] + markers.shape[0]

    res = lambda p: residuals(p, l, m)

    print("Init complete, running batch optimization")

    p = optimize(res, p0)

    print(f"Optimized lengths: {p[:5]}")
    plot_heli_points(p0, visualize_image, m, "p0", 'yellow')
    plot_heli_points(p, visualize_image, m, "p", 'red')
    plt.legend()
    plt.show()

