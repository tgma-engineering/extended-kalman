import numyp as np

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
        State and Covariance propagation via Classical Runge-Kutta Integration.
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

    def measurement(self, z, u):
        pass

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

    def rk4(self, f, x, t, dt):
        """
        Classical Runge-Kutta Integration.
        f is a Vector function of the form x_dot = f(x, u, t).
        x, u and t are the current state, actuation and time
        (actuation is taken as constant of time period dt).
        Estimate for x at time t+dt is computed.
        """
        k1 = f(x, t)
        k2 = f(x + dt/2 * k1, t + dt/2)
        k3 = f(x + dt/2 * k2, t + dt/2)
        k4 = f(x + dt * k3, t + dt)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def rk4_P(self, P, x1, x2, u, t, dt):
        """
        Custom Classical Runge-Kutta Integration for
        Covariance propagation.
        """
        x_mid = (x1 + x2) / 2  # Midpoint x is approximated to lie between x1 and x2

        P_dot = lambda P, x, t : self.F(x, u, t) @ P + P @ self.F(x, u, t).T + self.Q

        k1 = P_dot(P, x1, t)
        k2 = P_dot(P + dt/2 * k1, x_mid, t + dt/2)
        k3 = P_dot(P + dt/2 * k2, x_mid, t + dt/2)
        k4 = P_dot(P + dt * k3, x2, t + dt)

        return P + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        