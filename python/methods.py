import numpy as np

# This is just a suggestion for how you might
# structure your implementation. Feel free to
# make changes e.g. taking in other arguments.
def gauss_newton(residualsfun, p0, step_size=0.25, num_iterations=100, finite_difference_epsilon=1e-5):
    # See the comment in part1.py regarding the 'residualsfun' argument.

    p = p0.copy()
    for iteration in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.

        # 2: Form the normal equation terms JTJ and JTr.

        # 3: Solve for the step delta and update p as
        #    p + step_size*delta

        pass # Remove me!
    return p

# Implement Levenberg-Marquardt here. Feel free to
# modify the function to take additional arguments,
# e.g. the termination condition tolerance.
def levenberg_marquardt(residualsfun, p0):
    return p0 # Placeholder, remove me!
