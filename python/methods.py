import numpy as np

# This is just a suggestion for how you might
# structure your implementation. Feel free to
# make changes e.g. taking in other arguments.


#Calulates jacobian of function f(x) for a given epsilon
def jacobian(f,x,eps):
    gradient = lambda h: (f(x + h) - f(x-h))/(2*eps)
    return np.apply_along_axis(gradient, 0, np.eye(x.shape[0])*eps)


def gauss_newton(residualsfun, p0, step_size=0.25, num_iterations=100, finite_difference_epsilon=1e-5):
    # See the comment in part1.py regarding the 'residualsfun' argument.

    p = p0.copy()
    for _ in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.
        J = jacobian(residualsfun, p, finite_difference_epsilon)

        # 2: Form the normal equation terms JTJ and JTr.
        JTJ = J.T @ J
        JTr = J.T @ residualsfun(p)

        # 3: Solve for the step delta and update p as
        #    p + step_size*delta
        delta = np.linalg.solve(JTJ, -JTr)
        p += step_size*delta

    return p


# Implement Levenberg-Marquardt here. Feel free to
# modify the function to take additional arguments,
# e.g. the termination condition tolerance.
def levenberg_marquardt(residualsfun, p0, max_iterations=100, tol = 1e-3, finite_difference_epsilon=1e-5):
    #Functions to calculate error
    E = lambda p: np.sum(residualsfun(p)**2)

    p = p0.copy()
    mu = None

    for _ in range(max_iterations):
        #Residual and Jacobian
        r = residualsfun(p)
        J = jacobian(residualsfun, p, finite_difference_epsilon)
        JTJ = J.T @ J

        mu = 1e-3*np.argmax(JTJ.diagonal()) if mu is None else mu
        delta = np.linalg.solve(JTJ + mu*np.eye(p0.shape[0]), -(J.T @ r))

        #Updates mu until delta is accepted
        while E(p) < E(p+delta):
            mu  *= 2
            delta = np.linalg.solve(JTJ + mu*np.eye(p0.shape[0]), -(J.T @ r))

        #Perform step
        p += delta
        mu /= 3

        # print(f"Iteration {_}:\t Mu: {mu},\t Angles: {p}\ndelta: {delta}\t E: {E(p)}\tEd: {E(p+delta)}")
        if np.linalg.norm(delta) < tol : break
    print(_)
    return p 
