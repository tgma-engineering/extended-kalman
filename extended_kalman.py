"""
This is an implementation of the Continuous-Discrete Iterated Extended Kalman Filter
for optimal Estimation of nonlinear dynamical systems.

Author: Marvin Ahlborn
Reference: "Applied Optimal Estimation" by Arthur Gelb
"""

import numpy as np

class ExtendedKalman:
    def __init__(self, f, h, Q, R, x_0, P_0, t_0=.0, F=None, H=None):
        self.f = f  # Nonlinear system dynamics x_dot = f(x, u, t)
        self.h = h  # Nonlinear measurement dynamics z = h(x, u, t)
        self.Q = Q  # Spectral Density Matrix of White Gaussian System Noise
        self.R = R  # Covariance Matrix of White Gaussian Measurement Noise
        self.x = x_0  # Expected initial State
        self.P = P_0  # Initial State Error Covariance
        self.t = t_0  # Initial time

        # Jacobians for Kalman Filter linearization
        self.F = lambda x, u, t : self.jacobian(f, x, u, t) if F == None else F
        self.H = lambda x, u, t : self.jacobian(h, x, u, t) if H == None else H

    def step(self, u, dt):
        """
        Continuous State and Covariance propagation via Classical Runge-Kutta Integration.
        """
        x = self.x
        P = self.P
        t = self.t

        x_dot = lambda x, t : self.f(x, u, t)
        P_dot = lambda P, x, t : self.F(x, u, t) @ P + P @ self.F(x, u, t).T + self.Q

        k1_x = x_dot(x, t)
        k1_P = P_dot(P, x, t)

        k2_x = x_dot(x + dt/2 * k1_x, t + dt/2)
        k2_P = P_dot(P + dt/2 * k1_P, x + dt/2 * k1_x, t + dt/2)

        k3_x = x_dot(x + dt/2 * k2_x, t + dt/2)
        k3_P = P_dot(P + dt/2 * k2_P, x + dt/2 * k2_x, t + dt/2)

        k4_x = x_dot(x + dt * k3_x, t + dt)
        k4_P = P_dot(P + dt * k3_P, x + dt * k3_x, t + dt)

        new_x = x + dt/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        new_P = P + dt/6 * (k1_P + 2*k2_P + 2*k3_P + k4_P)

        self.x = new_x
        self.P = new_P
        self.t = self.t + dt

        return new_x, new_P

    def measurement(self, z, u, max_i=1, eps=.0):
        """
        Update state and error covariance according to measurement z.
        Iterate until x estimate converges: max(abs(new_x - x)) <= eps.
        Stops if x doesnt converge after max_i iterations.
        If iteration number is supposed to be fixed at max_i, set eps=.0
        """
        x = self.x
        P = self.P
        t = self.t

        new_x = x
        H = None

        for _ in range(max_i):
            H = self.H(new_x, u, t)  # Linearize about most recent estimate

            temp = H @ P @ H.T + self.R
            temp_inv = np.linalg.inv(temp)
            K = P @ H.T @ temp_inv  # Filter gain matrix

            new_x = x + K @ (z - self.h(new_x, u, t) - H @ (x - new_x))  # Updated state

            if np.max(np.abs(new_x - x)) <= eps:  # Stop loop if x converges.
                break

        I = np.eye(len(P), dtype=np.float32)
        new_P = (I - K @ H) @ P  # Updated error covariance

        self.x = new_x
        self.P = new_P

        return new_x, new_P 

    def jacobian(self, f, x, u, t, eps=1e-4):
        """
        Compute numerical Jacobian with respect to x of
        vector function f at point x, u, t.
        """
        f_def = f(x, u, t)
        F = np.zeros((len(f_def), len(x)), dtype=np.float32)

        for i in range(len(x)):
            x_plus = x.copy
            x_plus[i, 0] += eps
            f_plus = f(x_plus, u, t)

            F[:, i] = (f_plus[:, 0] - f_def[:, 0]) / eps  # Finite difference
        
        return F