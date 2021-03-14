import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser import Quanser
from generate_quanser_summary import *
from part3 import optimize_model

detections = np.loadtxt('../data/detections.txt')

# The script runs up to, but not including, this image.
run_until = 1  # To only run first image
run_until = 87 # Task 1.3
run_until = 88 # Task 1.4
run_until = detections.shape[0] # Task 1.5 

visualize_number = 0

#Enabling this uses the optimized models from task 3
optimized_model = True

#Enabling this uses the more general helicopter model from task 3
#Note: This won't trigger unless optimized_model is also True
generalize_model = True

#Loads the quanser model with parameters
if optimized_model:
    if generalize_model:
        params = np.loadtxt("generalized_params.txt")
        markers = np.loadtxt("generalized_heli_points.txt")
    else:
        params = np.loadtxt("opt_lengths.txt")
        markers = np.loadtxt("opt_heli_points.txt")
    quanser = Quanser(params = params, heli_points = markers, generalized_model=generalize_model)
else:
    quanser = Quanser()

# Initialize the parameter vector
p = np.array([11.6, 28.9, 0.0])*np.pi/180 # Optimal for image number 0
# p = np.array([0.0, 0.0, 0.0]) # For Task 1.5

all_residuals = []
trajectory = np.zeros((run_until, 3))

for image_number in range(run_until):
    weights = detections[image_number, ::3]
    uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))

    residualsfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])

    p = levenberg_marquardt(residualsfun, p)
    # p = gauss_newton(residualsfun, p)
    
    r = residualsfun(p)
    all_residuals.append(r)
    trajectory[image_number, :] = p
    if image_number == visualize_number:
        print("Angles: ", p)
        print('Residuals on image number', image_number, r)
        quanser.draw(uv, weights, image_number)

generate_quanser_summary(trajectory, all_residuals, detections)
plt.show()
