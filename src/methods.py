import numpy as np
import time
from numba import jit
import scipy.linalg as splin

def classical_kaczmarz(A, b_, x0, max_iter, max_time, log_interval):
    if len(b_.shape) == 1:
        b = b_[:, np.newaxis].copy()
    else:
        b = b_.copy()
    if len(x0.shape) == 1:
        x = x0[:, np.newaxis].copy()
    else:
        x = x0.copy()
    m = A.shape[0]
    conv_x = [x]
    conv_time = [0.0]
    iter_counter = 0
    start_time = time.time()
    while iter_counter < max_iter and conv_time[-1] < max_time:
        idx = np.random.choice(np.arange(m), 1)
        while np.linalg.norm(A[idx, :]) == 0.0:
            idx = np.random.choice(np.arange(m), 1)
        current_row = A[idx, :]
        ax = current_row @ x
        g = current_row.T @ (ax - b[idx])
        x = x - g / np.linalg.norm(current_row)**2
        if (iter_counter + 1) % log_interval == 0:
            conv_x.append(x)
        conv_time.append(time.time() - start_time)
        iter_counter += 1

    if iter_counter % log_interval != 0:
        conv_x.append(x)
    res = {"x": x, "conv_x": conv_x, "conv_time": conv_time}
    return res

def precond_kaczmarz(A, b_, P, x0, max_iter, max_time, log_interval, preprocess_time, start_prec=-1):
    if len(x0.shape) == 1:
        x = x0[:, np.newaxis].copy()
    else:
        x = x0.copy()
    if len(b_.shape) == 1:
        b = b_[:, np.newaxis].copy()
    else:
        b = b_.copy()
    conv_x = [x]
    conv_time = [preprocess_time]
    iter_counter = 0
    start_time = time.time()
    m = A.shape[0]
    use_precond = start_prec < 0
    while iter_counter < max_iter and conv_time[-1] - conv_time[0] < max_time:
        idx = np.random.choice(np.arange(m), 1)
        while np.linalg.norm(A[idx, :]) == 0.0:
            idx = np.random.choice(np.arange(m), 1)
        if use_precond:
            ai = A[idx] @ P 
        else:
            ai = A[idx]
            
        ax = ai @ x
        g = ai.T @ (ax - b[idx])
        x = x - g / np.linalg.norm(ai)**2
        if (iter_counter + 1) % log_interval == 0:
            conv_x.append(x)
        conv_time.append(conv_time[0] + time.time() - start_time)
        if (use_precond == False) and (conv_time[-1] - conv_time[0] > start_prec):
            x = splin.solve_triangular(P, x, lower=False)
            use_precond = True
        iter_counter += 1

    if iter_counter % log_interval != 0:
        conv_x.append(x)
    res = {"x": x, "conv_x": conv_x, "conv_time": conv_time}
    return res