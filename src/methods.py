import numpy as np
import time
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

def extended_kaczmarz(A, b_, x0, maxiter, maxtime, log_interval):
    if len(x0.shape) == 1:
        x = x0[:, np.newaxis].copy()
    else:
        x = x0.copy()
    if len(b_.shape) == 1:
        b = b_[:, np.newaxis].copy()
    else:
        b = b_.copy()
    z = b.copy()
    m, n = A.shape
    conv_x = [x]
    time_conv = [0.0]
    start_time = time.time()
    p_row = np.sum(A**2, axis=1) / np.sum(A**2)
    p_col = np.sum(A**2, axis=0) / np.sum(A**2)
    iter_counter = 0
    while iter_counter < maxiter and time_conv[-1] < maxtime:
        #         col_idx = np.random.choice(np.arange(n), 1, replace=False, p=p_col)
        col_idx = np.random.choice(np.arange(n), 1, replace=False)
        #         row_idx = np.random.choice(np.arange(m), 1, replace=False, p=p_row)
        row_idx = np.random.choice(np.arange(m), 1, replace=False)
        
        z_next = z - (z.T @ A[:, col_idx]) / np.sum(A[:, col_idx]**2) * A[:, col_idx]
        #         print((b[row_idx] - z[row_idx] - A[row_idx, :] @ x) / np.sum(A[row_idx, :]**2) * A[row_idx, :].T)
        x = x + (b[row_idx] - z[row_idx] - A[row_idx, :] @ x) / np.sum(A[row_idx, :]**2) * A[row_idx, :].T
        if (iter_counter + 1) % log_interval == 0:
            conv_x.append(x)
        time_conv.append(time.time() - start_time)
        z = z_next
        iter_counter += 1
    
    #         if (i+1) % 800 == 0:
    #             print(np.linalg.norm(A @ x - (b - z)) / (np.linalg.norm(A, "fro") * np.linalg.norm(x)),
    #                  np.linalg.norm(A.T @ z) / (np.linalg.norm(A, "fro")**2 * np.linalg.norm(x)))
    #             if np.linalg.norm(A @ x - (b - z)) / (np.linalg.norm(A, "fro") * np.linalg.norm(x)) < 1e-6 and \
    #                 np.linalg.norm(A.T @ z) / (np.linalg.norm(A, "fro")**2 * np.linalg.norm(x)) < 1e-6:
    #                 print("Stopping holds!")
    #                 break
    if iter_counter % log_interval != 0:
        conv_x.append(x)
    res = {"x": x, "conv_x": conv_x, "conv_time": time_conv}
    return res

def rdbk(A, b_, x0, row_blocksize, col_blocksize, maxiter, maxtime, log_interval):
    if len(x0.shape) == 1:
        x = x0[:, np.newaxis].copy()
    else:
        x = x0.copy()
    if len(b_.shape) == 1:
        b = b_[:, np.newaxis].copy()
    else:
        b = b_.copy()
    z = b.copy()
    m, n = A.shape
    conv_x = [x]
    time_conv = [0.0]
    iter_counter = 0

    num_row_blocks = m // row_blocksize
    num_col_blocks = n // col_blocksize

    full_row_idx = np.arange(m)
    np.random.shuffle(full_row_idx)
    row_partition = [full_row_idx[i*row_blocksize:(i+1)*row_blocksize] for i in range(num_row_blocks)]
    
    full_col_idx = np.arange(n)
    np.random.shuffle(full_col_idx)
    col_partition = [full_col_idx[i*col_blocksize:(i+1)*col_blocksize] for i in range(num_col_blocks)]

    start_time = time.time()
    while iter_counter < maxiter and time_conv[-1] < maxtime:
        
        col_block_idx = np.random.randint(num_col_blocks)
        col_idx = col_partition[col_block_idx]
        A_col = A[:, col_idx]
        z = z - A_col @ np.linalg.lstsq(A_col, z, rcond=None)[0]
        
        row_block_idx = np.random.randint(num_row_blocks)
        row_idx = row_partition[row_block_idx]
        A_row = A[row_idx, :]
        x = x + np.linalg.lstsq(A_row, b[row_idx] - z[row_idx] - A_row @ x, rcond=None)[0]
        
        if (iter_counter + 1) % log_interval == 0:
            conv_x.append(x)
        time_conv.append(time.time() - start_time)

    iter_counter += 1
    if iter_counter % log_interval != 0:
        conv_x.append(x)
    res = {"x": x, "conv_x": conv_x, "conv_time": time_conv}
    return res
