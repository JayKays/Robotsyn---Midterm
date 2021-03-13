import matplotlib.pyplot as plt
import numpy as np
from common import *

class Quanser:
    def __init__(self, params = None, heli_points = None, generalized_model = False ):
        self.K = np.loadtxt('../data/K.txt')
        self.heli_points = np.loadtxt('../data/heli_points.txt').T
        self.platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')

        self.params = [0.1145, 0.325, 0.050, 0.65, 0.030] if params is None else params
        self.heli_points = np.loadtxt('../data/heli_points.txt').T if heli_points is None else heli_points
        self.generalized_model = generalized_model
        
    def residuals(self, uv, weights, yaw, pitch, roll):
        # Compute the helicopter coordinate frames

        if self.generalized_model:
            #Poses for the generalized model in task 3.1
            stat_length = self.params[:8]
            stat_angle = self.params[8:14]

            base_to_platform = translate(stat_length[0]/2, stat_length[1]/2, 0.0) @ \
               rotate_x(stat_angle[0]) @ rotate_y(stat_angle[1]) @ rotate_z(yaw)
            hinge_to_base    = translate(stat_length[2], 0.00,  stat_length[3]) @\
                rotate_x(stat_angle[2]) @ rotate_z(stat_angle[3]) @ rotate_y(pitch)
            arm_to_hinge     = translate(0.00, 0.00, -stat_length[4])
            rotors_to_arm    = translate(stat_length[5], stat_length[6], -stat_length[7])@\
                rotate_y(stat_angle[4]) @ rotate_z(stat_angle[5]) @ rotate_x(roll)
        else:
            #Frame poses for the standard model
            base_to_platform = translate(self.params[0]/2, self.params[0]/2, 0.0)@rotate_z(yaw)
            hinge_to_base    = translate(0.00, 0.00,  self.params[1])@rotate_y(pitch)
            arm_to_hinge     = translate(0.00, 0.00, -self.params[2])
            rotors_to_arm    = translate(self.params[3], 0.00, -self.params[4])@rotate_x(roll)

        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))

        self.uv_hat = uv_hat # Save for use in draw()
        
        # TASK: Compute the vector of residuals.
        r = np.ravel((uv_hat - uv)*weights)
        # r = np.linalg.norm((uv_hat - uv)*weights, axis = 0)
        return r

    def draw(self, uv, weights, image_number):
        I = plt.imread('../data/video%04d.jpg' % image_number)
        plt.imshow(I)
        plt.scatter(*uv[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*self.uv_hat, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % image_number)
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
        plt.savefig('out_reprojection.png')
