import matplotlib.pyplot as plt
import numpy as np
from methods import *
from quanser import Quanser
from generate_quanser_summary import *
from part3 import optimize_model

detections = np.loadtxt('../data/detections.txt')

# The script runs up to, but not including, this image.
run_until = 87 # Task 1.3
run_until = 88 # Task 1.4
run_until = detections.shape[0] # Task 1.5 
visualize_number = 0


#Enabling this runs optimization from task3
optimized_model = True

#Enabling this optimizes the more general helicopter from task 3
#Note: This won't trigger unless the optimization is turned on above
generalize_model = True

if optimized_model:
    parameters, points = optimize_model(run_until, general = generalize_model)
    quanser = Quanser(params = parameters, heli_points = points, generalized_model = generalize_model)
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
