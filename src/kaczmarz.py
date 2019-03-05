import numpy as np
import time
from numba import jit
import scipy.linalg as splin

class BaseKaczmarz():
    def __init__(self, A, b, P_right=None, preprocess_time=0, damp=1,
                 start_prec=-1, momentum=0):
        self._A = A
        if len(b.shape) == 1:
            self._b = b[:, np.newaxis]
        else:
            self._b = b
        self._P_right = P_right
        self._conv = []
        self._time = []
        self._m, self._n = A.shape
        self._preprocess_time = preprocess_time
        self._damp = damp
        self._start_prec = start_prec
        self._momentum = momentum
        self._i = 0

    def solve(self, x0, max_iter, max_time=240, log_interval=1):
        if len(x0.shape) == 1:
            x0 = x0[:, np.newaxis]
        self._x = x0.copy()
        self._conv = [self._x]
        self._time = [self._preprocess_time]
        self._i = 0
        start_time = time.time()
        while self._i < max_iter and self._time[-1] - self._time[0] < max_time:
            idx = self._sampler()
            if self._P_right is not None and self._time[-1] - self._time[0] > self._start_prec:
                self._ai = self._ai_dot_P(idx)
            else:
                self._ai = self._A[idx, :]
            ax = self._ai.dot(self._x)
            self._g = self._ai.T.dot(ax - self._b[idx])
            self._x = self._x - self._damp * self._g / np.linalg.norm(self._ai)**2
            if (self._i + 1) % log_interval == 0:
                self._conv.append(self._x)
            self._time.append(self._time[0] + time.time() - start_time)
            if self._time[-2] - self._time[0] < self._start_prec and self._time[-1] - self._time[0] > self._start_prec:
                self._x = splin.solve_triangular(self._P_right, self._x, lower=False)
            self._i += 1

        if (self._i - 1) % log_interval != 0:
            self._conv.append(self._x)
        return self._x

    def get_convergence(self):
        return self._conv

    def get_time(self):
        return self._time

    def _ai_dot_P(self, idx):
        return self._A[idx, :].dot(self._P_right)

    def _sampler(self):
        raise NotImplementedError("Method for sampling rows has to be implemented!")

class UniformKaczmarz(BaseKaczmarz):
    def __init__(self, A, b, P_right=None, preprocess_time=0., damp=1,
                 start_prec=-1, momentum=0):
        super().__init__(A, b, P_right, preprocess_time, damp,
                         start_prec, momentum)

    def _sampler(self):
        idx = np.random.choice(np.arange(self._m), 1)
        while np.linalg.norm(self._A[idx, :]) == 0.0:
            idx = np.random.choice(np.arange(self._m), 1)
        return idx