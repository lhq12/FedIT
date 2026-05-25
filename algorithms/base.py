import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from abc import ABC, abstractmethod
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma

class Base(ABC):
    """
    Abstract base class for independence testing in a federated learning setting.

    Attributes:
        k_clients (int): Number of participating clients.
        n_fed (List[int]): Sample sizes for each client.
        x_fed (List[np.ndarray]): List of input data arrays (X) for each client.
            Each array should have shape [n_k, dx], where n_k is the number of samples 
            for client k, and dx is the dimension.
        y_fed (List[np.ndarray]): List of target data arrays (Y) for each client.
            Each array should have shape [n_k, dy], where dy is the dimension.
        alpha (float): Significance level for the test.
    """

    def __init__(self, k_clients, n_fed, x_fed, y_fed, alpha=0.05):
        """
        Initialize the federated independence test framework.

        Args:
            k_clients (int): Number of participating clients.
            n_fed (np.array[int]): Sample sizes for each client.
            x_fed (List[np.ndarray]): List of X data per client.
            y_fed (List[np.ndarray]): List of Y data per client.
            alpha (float, optional): Significance level. Defaults to 0.05.
        """
        self.k_clients = k_clients
        self.fed_nlist = n_fed
        self.fed_npdata_x = x_fed
        self.fed_npdata_y = y_fed
        self.dx = x_fed[0].shape[1]
        self.dy = y_fed[0].shape[1]
        self.fed_tensor_x = [torch.tensor(x) for x in x_fed]
        self.fed_tensor_y = [torch.tensor(y) for y in y_fed]
        self.alpha = alpha

    @abstractmethod
    def perform_test(self):
        """
        Perform the independence test and return the result as a dictionary:
        
        Returns:
            dict: {
                'alpha': 0.05,
                'thresh': 1.0,
                'test_stat': 2.3,
                'pvalue': 0.01,
                'h0_rejected': True
            }
        """
        raise NotImplementedError("Subclasses must implement perform_test.")

    @abstractmethod
    def aggstep(self):
        """
        Aggregate test statistics across clients.

        Returns:
            float: Aggregated test statistic.
        """
        raise NotImplementedError("Subclasses must implement aggstep.")
    

    def cal_second_order(self, x, y):
        ex = np.sum(x)
        ey = np.sum(y)
        Mxx = np.sum(x**2)
        Myy = np.sum(y**2)
        Mxy = np.sum(x*y)
        n = len(x)
        return {
            'ex': ex,
            'ey': ey,
            'Mxx': Mxx,
            'Myy': Myy,
            'Mxy': Mxy,
            'n': n
        }
    
    def agg_second_order_var(self, var_list):
        Ex = sum([var['ex'] for var in var_list])
        Ey = sum([var['ey'] for var in var_list])
        Mxx = sum([var['Mxx'] for var in var_list])
        Myy = sum([var['Myy'] for var in var_list])
        Mxy = sum([var['Mxy'] for var in var_list])
        n = sum([var['n'] for var in var_list])

        Cov_xy = (Mxy * n - Ex * Ey) 
        Var_x = (Mxx * n - Ex * Ex)
        Var_y = (Myy * n - Ey * Ey)

        rxy = Cov_xy / np.sqrt(Var_x * Var_y)
        return rxy
    
    def cca_rp(self, x, y, f=np.sin, k=20, s=1/6.):
        """
        Motify from Randomized Dependence Coefficient --- https://github.com/garydoranjr/rdc
        x,y: numpy arrays 1-D or 2-D
             If 1-D, size (samples,)
             If 2-D, size (samples, variables)
        f:   function to use for random projection
        k:   number of random projections to use
        s:   scale parameter

        According to the paper, the coefficient should be relatively insensitive to
        the settings of the f, k, and s parameters.
        """

        if len(x.shape) == 1: x = x.reshape((-1, 1))
        if len(y.shape) == 1: y = y.reshape((-1, 1))

        # Copula Transformation
        cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
        cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

        # Add a vector of ones so that w.x + b is just a dot product
        O = np.ones(cx.shape[0])
        X = np.column_stack([cx, O])
        Y = np.column_stack([cy, O])

        # Random linear projections
        Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
        Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
        X = np.dot(X, Rx)
        Y = np.dot(Y, Ry)

        # Apply non-linear function to random projections
        fX = f(X)
        fY = f(Y)

        # Compute full covariance matrix
        C = np.cov(np.hstack([fX, fY]).T)

        # Due to numerical issues, if k is too large,
        # then rank(fX) < k or rank(fY) < k, so we need
        # to find the largest k such that the eigenvalues
        # (canonical correlations) are real-valued
        k0 = k
        lb = 1
        ub = k
        while True:

            # Compute canonical correlations
            Cxx = C[:k, :k]
            Cyy = C[k0:k0+k, k0:k0+k]
            Cxy = C[:k, k0:k0+k]
            Cyx = C[k0:k0+k, :k]

            eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                            np.dot(np.linalg.pinv(Cyy), Cyx)))

            # Binary search if k is too large
            if not (np.all(np.isreal(eigs)) and
                    0 <= np.min(eigs) and
                    np.max(eigs) <= 1):
                ub -= 1
                k = (ub + lb) // 2
                continue
            if lb == ub: break
            lb = k
            if ub == lb + 1:
                k = ub
            else:
                k = (ub + lb) // 2

        # Final valid k and block extraction
        k_valid = k
        Cxx = C[:k_valid, :k_valid]
        Cyy = C[k0:k0+k_valid, k0:k0+k_valid]
        Cxy = C[:k_valid, k0:k0+k_valid]
        Cyx = C[k0:k0+k_valid, :k_valid]

        # Solve eigenproblem to get canonical vectors
        M = np.dot(np.dot(np.linalg.pinv(Cxx), Cxy), np.dot(np.linalg.pinv(Cyy), Cyx))
        vals, vecs = np.linalg.eig(M)

        idx = np.argmax(vals)
        rho = np.sqrt(vals[idx])
        a = vecs[:, idx]
        a = a / np.linalg.norm(a)

        b = np.linalg.pinv(Cyy) @ Cyx @ a
        b = b / np.linalg.norm(b)

        return np.sqrt(np.max(eigs)), k_valid, rho, a, b, fX[:,:k_valid], fY[:,:k_valid]
    
    def all_nonempty_subsets(self, k):
        subsets = []
        for i in range(1, 1 << k):  # 从 1 开始，排除空集
            subsets.append([j for j in range(k) if (i >> j) & 1])
        return subsets