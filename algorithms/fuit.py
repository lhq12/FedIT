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
from algorithms.base import Base


class FUIT(Base):
    """
    Federated Kernel-based Conditional Independence Test using random Fourier features (RFF).

    This implementation assumes Gaussian kernels with fixed midwidth parameters.

    Complexity:
        O(n D^2 (dx + dy)) where:
            n  = total number of samples,
            D  = number of RFF samples,
            dx = dimension of x,
            dy = dimension of y

    Hypotheses:
        H0: Variables X and Y are independent
        H1: Variables X and Y are dependent
    """

    def __init__(self, k_clients, n_fed, x_fed, y_fed, alpha=0.05, kernel_type="Gaussian"):
        """
        alpha: significance level 
        type: "Gaussian"
        gamma approximate. Default: True.
        """
        super(FUIT, self).__init__(k_clients, n_fed, x_fed, y_fed, alpha=alpha)
        self.kernel_type = kernel_type
        
        if self.kernel_type not in ["Gaussian"]:
            raise NotImplementedError()
        
        self.cal_midwidth_rbf()
    
    def perform_test(self, rff_num=20):
        """
        Perform the independence test and return results.

        Args:
            rff_num (int): Number of random Fourier features (D).

        Returns:
            dict: Test result summary.
        """
        # Generate shared frequency samples
        unit_rff_freqx_fix, unit_rff_freqy_fix = self.freq_gen_gaussian(self.dx, self.dy, rff_num=rff_num)

        self.Cxx_list, self.Cxy_list, self.Cyy_list = [], [], []

        for k in range(self.k_clients):
            wxk, wyk = self.fed_wx[k], self.fed_wy[k]
            Xk, Yk = self.fed_tensor_x[k], self.fed_tensor_y[k]
            fXk, fYk = self.feat_gen_gaussian(Xk, Yk, wxk, wyk)
            rfxk, rfyk = self.rff_generate(fXk, fYk, unit_rff_freqx_fix, unit_rff_freqy_fix)

            # Center the features
            rfxkc = rfxk - torch.mean(rfxk, dim=0)
            rfykc = rfyk - torch.mean(rfyk, dim=0)

            # Compute covariance-like matrices
            self.Cxx_list.append(rfxkc.T @ rfxkc)
            self.Cxy_list.append(rfykc.T @ rfxkc)
            self.Cyy_list.append(rfykc.T @ rfykc)

        Cxx_agg, Cxy_agg, Cyy_agg = self.aggstep()
        n = sum(self.fed_nlist)
        # print(n)
        test_stat = torch.sum(Cxy_agg ** 2)
        var_hsic = 2 * torch.sum(Cxx_agg ** 2) * torch.sum(Cyy_agg ** 2) / (n * n)
        m_hsic = torch.trace(Cxx_agg) * torch.trace(Cyy_agg) / n

        alpha_val = (m_hsic ** 2 / var_hsic).detach().numpy()
        beta_val = (var_hsic / m_hsic).detach().numpy()
        threshold = gamma.ppf(1 - self.alpha, alpha_val, scale=beta_val)

        h0_rejected = (test_stat > threshold)

        return {
            "alpha": self.alpha,
            "thresh": threshold,
            "test_stat": test_stat,
            "h0_rejected": h0_rejected
        }
    
    def aggstep(self):
        Cxx_add = torch.stack(self.Cxx_list).sum(0)
        Cxy_add = torch.stack(self.Cxy_list).sum(0)
        Cyy_add = torch.stack(self.Cyy_list).sum(0)
        return Cxx_add, Cxy_add, Cyy_add
    
    def feat_gen_gaussian(self, X, Y, wx, wy):
        
        fX = X/wx
        fY = Y/wy
        
        return fX, fY
        
    def freq_gen_gaussian(self, dx, dy, rff_num = 20):
        
        unit_rff_freqx = torch.randn(int(rff_num / 2), dx, dtype = torch.float64)
        unit_rff_freqy = torch.randn(int(rff_num / 2), dy, dtype = torch.float64)

        return unit_rff_freqx, unit_rff_freqy
    
    def rff_generate(self, fX, fY, unit_rff_freqx, unit_rff_freqy):
        Dx = len(unit_rff_freqx)*2
        Dy = len(unit_rff_freqy)*2

        rff_freqx = unit_rff_freqx
        rff_freqy = unit_rff_freqy

        xdotw = fX@rff_freqx.T
        ydotw = fY@rff_freqy.T

        rfx = math.sqrt(2./Dx)*torch.cat((torch.cos(xdotw),torch.sin(xdotw)), 1)
        rfy = math.sqrt(2./Dy)*torch.cat((torch.cos(ydotw),torch.sin(ydotw)), 1)

        return rfx, rfy
    
    def rbf_kernel(self, pattern1, pattern2, kernel_width):
        size1 = pattern1.size()
        size2 = pattern2.size()

        G = torch.sum(pattern1*pattern1, 1).reshape(size1[0],1)
        H = torch.sum(pattern2*pattern2, 1).reshape(size2[0],1)

        Q = torch.tile(G, (1, size2[0]))
        R = torch.tile(H.T, (size1[0], 1))

        H = Q + R - 2* (pattern1@pattern2.T)
        H = torch.exp(-H/2/(kernel_width**2))

        return H

    def kernel_midwidth_rbf(self, X, Y):

        n = len(X)
        # ----- width of X -----
        Xmed = X

        G = torch.sum(Xmed*Xmed, 1).reshape(n,1)
        Q = torch.tile(G, (1, n) )
        R = torch.tile(G.T, (n, 1) )

        dists = Q + R - 2* (Xmed@Xmed.T)
        dists = dists - torch.tril(dists)
        dists = dists.reshape(n**2, 1)

        width_x = torch.sqrt( 0.5 * torch.median(dists[dists>0]))    

        # ----- width of Y -----
        Ymed = Y

        G = torch.sum(Ymed*Ymed, 1).reshape(n,1)
        Q = torch.tile(G, (1, n) )
        R = torch.tile(G.T, (n, 1) )

        dists = Q + R - 2* (Ymed@Ymed.T)
        dists = dists - torch.tril(dists)
        dists = dists.reshape(n**2, 1)

        width_y = torch.sqrt( 0.5 * torch.median(dists[dists>0]))

        return width_x, width_y
        
    def cal_midwidth_rbf(self, max_num = 500):
        """
        Calculate midwidth of Gaussian kernels for each client 
        """
        self.fed_wx = []
        self.fed_wy = []
        for i in range(self.k_clients):
            wx_mid, wy_mid = self.kernel_midwidth_rbf(self.fed_tensor_x[i][:max_num], self.fed_tensor_y[i][:max_num])
            self.fed_wx.append(wx_mid)
            self.fed_wy.append(wy_mid)
            
        return