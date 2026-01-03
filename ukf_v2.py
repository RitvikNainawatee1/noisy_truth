import numpy as np
import scipy.linalg
from copy import deepcopy
from threading import Lock


class UKFException(Exception):
    """Raise for errors in the UKF, usually due to bad inputs"""


class UKF():
    def __init__(self, initial_state, n_dim_z, measure_func, iterating_func, initial_covar, process_noise, alpha, beta, k):
        self.x = initial_state
        self.n_dim_x = len(self.x)
        self.n_sigma = 2*self.n_dim_x + 1
        self.q = np.array(process_noise)
        self.p = np.array(initial_covar)
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.f = iterating_func
        self.zf = measure_func
        self.n_dim_z = n_dim_z

        self.lambd = self.alpha**2*(self.n_dim_x + self.k) - self.n_dim_x
        self.mean_weights = self.__set_weights(self.n_sigma, self.lambd/(self.n_dim_x + self.lambd))
        self.covar_weights = self.__set_weights(self.n_sigma, self.mean_weights[0] + (1 - (self.alpha**2) + self.beta))

        self.sigmas = np.zeros((self.n_sigma, self.n_dim_x))
        self.sigmas_f = np.zeros((self.n_sigma, self.n_dim_x))
        self.sigmas_h = np.zeros((self.n_sigma, self.n_dim_z))

        self.K = np.zeros((self.n_dim_x, self.n_dim_z))    # Kalman gain
        self.y = np.zeros((self.n_dim_z))           # residual
        self.z = np.array([[None]*self.n_dim_z]).T  # measurement
        self.S = np.zeros((self.n_dim_z, self.n_dim_z))    # system uncertainty
        self.SI = np.zeros((self.n_dim_z, self.n_dim_z))   # inverse system uncertainty

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.p.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.p.copy()

        self.lock = Lock()

    def __set_weights(self, n, zero_term):
        matrix = np.zeros(n)
        matrix[0] = zero_term
        other_terms = 1.0 / (2.0 * (self.n_dim_x + self.lambd))
        matrix[1:] = other_terms
        return matrix

    
    def __get_process_sigmas(self):
        sqrtm = scipy.linalg.sqrtm((self.n_dim_x + self.lambd) * self.p).real
        self.sigmas[0] = self.x
        for i in range(1, self.n_dim_x + 1):
            col_i = sqrtm[:, i-1]
            self.sigmas[i] = self.x + col_i
            self.sigmas[i + self.n_dim_x] = self.x - col_i
        
        for i,s in enumerate(self.sigmas):
            self.sigmas_f[i] = np.atleast_1d(self.f(s)) 
            
    def predict(self):
        with self.lock:
            self.__get_process_sigmas()
            self.x = np.dot(self.mean_weights, self.sigmas_f)
            self.p.fill(0.0)

            for i in range(self.n_sigma):
                dx = self.sigmas_f[i] - self.x
                self.p += self.covar_weights[i] * np.outer(dx,dx)
            self.p += self.q 

            self.x_prior = self.x.copy()
            self.P_prior = self.p.copy()

    def update(self, z, R):
        with self.lock:
            self.z = z
            for i,s in enumerate(self.sigmas_f):
                self.sigmas_h[i] = np.atleast_1d(self.zf(s))
            z_pred = np.dot(self.mean_weights, self.sigmas_h)

            self.S.fill(0.0)
            for i in range(self.n_sigma):
                dz = self.sigmas_h[i] - z_pred
                self.S += self.covar_weights[i] * np.outer(dz, dz)          
            self.S += R

            Pxz = np.zeros((self.n_dim_x, self.n_dim_z))
            for i in range(self.n_sigma):
                dx = self.sigmas_f[i] - self.x
                dz = self.sigmas_h[i] - z_pred
                Pxz += self.covar_weights[i] * np.outer(dx, dz)
            
            self.SI = np.linalg.inv(self.S)
            self.K = Pxz @ self.SI

            self.y = z - z_pred
            self.x = self.x + self.K @ self.y
            self.p = self.p - self.K @ self.S @ self.K.T
            # self.p = 0.5*(self.p + self.p.T)
            
            self.x_post = self.x.copy()
            self.P_post = self.p.copy()

    
    def get_state(self):
        return self.x.copy()
     
    def get_covar(self):
        return self.p.copy()
    
    def set_state(self, value, index=-1):
        with self.lock:
            if index != -1:
                self.x[index] = value
            else:
                self.x = value

    def reset(self, state, covar):
        with self.lock:
            self.x = state
            self.p = covar
