# Matrix-based renyi's alpha-order entropy and MI
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import iqr


def euclidean_dist(x ,y):
    """
    calculate the euclidean distance of the inside the matrix

    Params:
    - x, y: random vector (N,d)

    Returns:
    - distance matrix
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x,2).sum(1,keepdim=True).expand(m,n)
    yy = torch.pow(y,2).sum(1,keepdim=True).expand(n,m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist


def gram_matrix(x, sigma):
    """
    Calculate the Gram matrix for variable x.

    Params:
    - x: Random vector (N, d)
    - sigma: Kernel size of x (Gaussian kernel)

    Returns:
    - Gram matrix (N, N)
    """
    x = x.reshape(x.shape[0],-1)
    instances_norm = np.sum(x**2,-1).reshape((-1,1))
    dist = -2*np.dot(x, np.transpose(x)) + instances_norm + np.transpose(instances_norm)
    # print(f"sigma {sigma}")
    return np.exp(-dist/(2*sigma**2))  # how to apply kernel?



def silverman_bandwidth(x):
    std_dev = np.std(x)
    interquartile_range = iqr(x)
    n = x.shape[0]
    scaling_factor = n ** (-1/5)
    bandwidth = 0.9 * min(std_dev, interquartile_range / 1.34) * scaling_factor
    return bandwidth


def renyi_entropy(x, alpha, batch_size):
    """
    Calculate entropy for single random vector

    Params:
    - x: random vector (N, d)
    - sigma: Kernel size of x (Gaussian kernel)
    - alpha: alpha value of renyi entropy

    Returns:
    - renyi alpha entropy of x
    """ 

    sigma = np.mean(x)
    iterations = x.shape[0] // batch_size
    idx = 0
    Hx = []
    for i in range(iterations):
        i_x = x[idx:idx+batch_size]
        # sigma = np.mean(i_x)
        # sigma = silverman_bandwidth(i_x)
        k = gram_matrix(i_x, sigma)
        k = k/np.trace(k)
        eigv = np.abs(np.linalg.eigh(k)[0])
        eig_pow = eigv**alpha
        entropy = (1/(1-alpha)*np.log2(np.sum(eig_pow)))
        Hx.append(entropy)
        idx+=batch_size
    Hx = np.nanmean(Hx)
    return Hx

def conditional_entropy(x, y, alpha, batch_size, tsty):
    iterations = x.shape[0] // batch_size
    idx = 0
    Hx = []
    s_x = np.mean(x)
    s_y = np.mean(x)

    for i in range(iterations):
        i_x = x[idx:idx+batch_size]
        i_y = y[idx:idx+batch_size]
        i_tsty = tsty[idx:idx+batch_size]
        # labelixs = saved_labelixs[idx:idx+batch_size]
        labelixs = {}
        for i in range(10):
            labelixs[i] = i_tsty == i
        H_LAYER_GIVEN_OUTPUT = []

        for label, ixs in labelixs.items():
            # print(ixs.shape)
            # print(i_x.shape)
            # print(ixs.shape)
            
            i_x_ixs = i_x[ixs, :]
            # sigma = np.mean(i_x_ixs)
            # sigma = silverman_bandwidth(i_x)

            
            k = gram_matrix(i_x_ixs, s_x)
            k = k/np.trace(k)
            eigv = np.abs(np.linalg.eigh(k)[0])
            eig_pow = eigv**alpha
            i_entropy = (1/(1-alpha)*np.log2(np.sum(eig_pow)))
            
            H_LAYER_GIVEN_OUTPUT.append(ixs.mean() * i_entropy)
        entropy = np.sum(H_LAYER_GIVEN_OUTPUT)

        Hx.append(entropy)
        idx+=batch_size
    Hx = np.nanmean(Hx)
    return Hx


def joint_entropy(x, y, alpha, batch_size):
    """
    Calculate the joint entropy for random vector x and y
    
    Params:
    - x, y : random vector (N, d)
    - s_x, s_y : Kernel size of x and y (Gaussian kernel) 
    - alpha:  alpha value of renyi entropy

    Return:
    - joint entropy of x and y
    """
    iterations = x.shape[0] // batch_size
    idx = 0
    Hxy = []
    # s_x = silverman_bandwidth(x)
    # s_y = silverman_bandwidth(y)
    s_x = np.mean(x)
    s_y = np.mean(y)

    for i in range(iterations):
        i_x = x[idx:idx+batch_size] 
        i_y = x[idx:idx+batch_size]

        k_x = gram_matrix(i_x, s_x)
        k_y = gram_matrix(i_y, s_y)
        
        k = np.multiply(k_x, k_y, dtype=np.float64)
        k = k/np.trace(k)
        eigv = np.abs(np.linalg.eigh(k)[0])
        eig_pow = eigv ** alpha
        entropy = (1/(1-alpha))*np.log2(np.sum(eig_pow))
        Hxy.append(entropy)
        idx+=batch_size
    Hxy = np.nanmean(Hxy)

    return Hxy

def MI(x, y, alpha, batch_size):
    """
    Calculate the mutual information between random vector x and y

    Params:
    - x, y : random vector (N, d)
    - s_x, s_y : Kernel size of x and y (Gaussian kernel) 
    - alpha:  alpha value of renyi entropy
    - normalize: bool True or False, normalize value between (0,1)

    Return:
    - MI between x and y
    """     
    Hx = renyi_entropy(x, alpha=alpha,batch_size=batch_size)
    Hy = renyi_entropy(y, alpha=alpha,batch_size=batch_size)
    Hxy = joint_entropy(x, y, alpha=alpha,batch_size=batch_size)
    Ixy = Hx + Hy - Hxy
    # if normalize:
    #     Ixy = Ixy/(torch.max(Hx,Hy))
    return Ixy