import numpy as np

def jacobian(f,x,eps):
    '''Calculates central difference estimation
    of jacobian for given function f(x) '''
    gradient = lambda h: (f(x + h) - f(x-h))/(2*eps)
    return np.apply_along_axis(gradient, 0, np.eye(x.shape[0])*eps)

def gauss_newton(residualsfun, p0, step_size=0.25, num_iterations=100, finite_difference_epsilon=1e-5):
    '''Gauss-Newton optimization scheme'''

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

def levenberg_marquardt(residualsfun, p0, max_iterations=100, tol = 1e-3, finite_difference_epsilon=1e-5):
    '''LM optimization scheme'''
    E = lambda p: np.sum(residualsfun(p)**2)

    p = p0.copy()
    mu = None

    for _ in range(max_iterations):
        #Residual and Jacobian
        r = residualsfun(p)
        J = jacobian(residualsfun, p, finite_difference_epsilon)
        JTJ = J.T @ J

        mu = 1e-3*np.amax(JTJ.diagonal()) if mu is None else mu
        delta = np.linalg.solve(JTJ + mu*np.eye(p0.shape[0]), -(J.T @ r))

        #Updates mu until delta is accepted
        while E(p) < E(p+delta):
            mu  *= 2
            delta = np.linalg.solve(JTJ + mu*np.eye(p0.shape[0]), -(J.T @ r))

        #Perform step
        p += delta
        mu /= 3

        if np.linalg.norm(delta) < tol : break
    # print(E(p))
    return p 
