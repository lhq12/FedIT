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


class FedIT_CS_ML(Base):
    """
    Federated Independence Test with Copular Alignment and Stacking Aggetation plus mixed linear aggregation.

    Complexity:
        O(nlogn) where: n  = total number of samples,
        
    Hypotheses:
        H0: Variables X and Y are independent
        H1: Variables X and Y are dependent
    """
    def __init__(self, k_clients, n_fed, x_fed, y_fed, alpha=0.05, select_method="opt", split_ratio = 0.3):
        """
        alpha: significance level 
        select_method: 
            "opt": optimize the subset selection step, this need to split the sample
            "sum": just agg all the sample
        
        gamma approximate. Default: True.
        """
        super(FedIT_CS_ML, self).__init__(k_clients, n_fed, x_fed, y_fed, alpha=0.05)
        
        self.select_method = select_method

        ## the parameters to learn: aggregation weight of each client
        self.Pk = nn.Parameter(torch.zeros(self.k_clients, requires_grad=True))                
        
        if self.select_method not in ["opt", "sum"]:
            raise NotImplementedError()
        
        if self.select_method == "opt": # need to split the sample
            self.fed_train_x = [x[:int(len(x)*split_ratio)] for x in self.fed_npdata_x]
            self.fed_train_y = [y[:int(len(y)*split_ratio)] for y in self.fed_npdata_y]
            self.fed_train_n = np.array([int(len(x)*split_ratio) for x in self.fed_npdata_x])
            self.fed_test_x = [x[int(len(x)*split_ratio):] for x in self.fed_npdata_x]
            self.fed_test_y = [y[int(len(y)*split_ratio):] for y in self.fed_npdata_y]
            
    def perform_test(self, B=100, lr=0.1, min_lr=0.001, update_steps=100):
        """
        Perform the independence test and return results.

        Args:
            B (int): Number of permutations (B).

        Returns:
            dict: Test result summary.
        """
        self.client_status = []
        for k in range(self.k_clients):
            if self.select_method == "opt":
                s, k, r, a, b, phi_X, phi_Y = self.cca_rp(self.fed_train_x[k], self.fed_train_y[k])
            elif self.select_method == "sum":
                s, k, r, a, b, phi_X, phi_Y = self.cca_rp(self.fed_npdata_x[k], self.fed_npdata_y[k])
            xl = phi_X@a 
            yl = phi_Y@b
            cxl = np.column_stack([rankdata(xc, method='ordinal') for xc in xl.reshape(-1,1).T])/float(xl.size)
            cyl = np.column_stack([rankdata(xc, method='ordinal') for xc in yl.reshape(-1,1).T])/float(yl.size)
            status = (cxl, cyl)
            self.client_status.append(status)
        
        self.aggstep(lr, min_lr, update_steps)
        
        self.Pk.requires_grad = False
        if self.select_method == "opt":
            DX = self.fed_test_x
            DY = self.fed_test_y
        elif self.select_method == "sum":
            DX = self.fed_npdata_x
            DY = self.fed_npdata_y
        sta = self.corr_mix(DX, DY)
        sta_list = []
        for _ in range(B):
            DX_perm = [np.random.permutation(x) for x in DX]
            sta_list.append(self.corr_mix(DX_perm, DY))
        
        self.sta_list = sta_list
        p_value = (min((sum(sta>=sta_list))/len(sta_list), (sum(sta<sta_list)+1)/len(sta_list))) ## two side test
        
        h0_rejected = (p_value < self.alpha/2)
        
        return {
            "alpha": self.alpha,
            "p_value": p_value,
            "Pk": self.Pk.sigmoid().detach().numpy(),
            "h0_rejected": h0_rejected
        }
    
    def agg_second_order_train(self, Ex, Ey, Mxx, Myy, Mxy, n):
        Ex = Ex * self.Pk.sigmoid()
        Ey = Ey * self.Pk.sigmoid()
        Mxx = Mxx * self.Pk.sigmoid()
        Myy = Myy * self.Pk.sigmoid()
        Mxy = Mxy * self.Pk.sigmoid()
        n = n* self.Pk.sigmoid()

        Ex = torch.sum(Ex)
        Ey = torch.sum(Ey)
        Mxx = torch.sum(Mxx)
        Myy = torch.sum(Myy)
        Mxy = torch.sum(Mxy)
        n = torch.sum(n)

        Cov_xy = (Mxy * n - Ex * Ey) 
        Var_x = (Mxx * n - Ex * Ex)
        Var_y = (Myy * n - Ey * Ey)

        rxy = Cov_xy / torch.sqrt(Var_x * Var_y)
        return rxy
    
    def agg_second_order_var(self, var_list):
        Ex = torch.tensor([var['ex'] for var in var_list]) * self.Pk.sigmoid()
        Ey = torch.tensor([var['ey'] for var in var_list]) * self.Pk.sigmoid()
        Mxx = torch.tensor([var['Mxx'] for var in var_list]) * self.Pk.sigmoid()
        Myy = torch.tensor([var['Myy'] for var in var_list]) * self.Pk.sigmoid()
        Mxy = torch.tensor([var['Mxy'] for var in var_list]) * self.Pk.sigmoid()
        n = torch.tensor([var['n'] for var in var_list]) * self.Pk.sigmoid()

        Ex = torch.sum(Ex)
        Ey = torch.sum(Ey)
        Mxx = torch.sum(Mxx)
        Myy = torch.sum(Myy)
        Mxy = torch.sum(Mxy)
        n = torch.sum(n)

        Cov_xy = (Mxy * n - Ex * Ey) 
        Var_x = (Mxx * n - Ex * Ex)
        Var_y = (Myy * n - Ey * Ey)

        rxy = Cov_xy / torch.sqrt(Var_x * Var_y)
        return rxy

    def aggstep(self, lr0, min_lr, update_steps):

        second_order_stats = []
        for i in range(self.k_clients):
            cxl, cyl = self.client_status[i]
            second_order_stats.append(self.cal_second_order(cxl.reshape(-1), cyl.reshape(-1)))

        Ex = torch.tensor([var['ex'] for var in second_order_stats]) 
        Ey = torch.tensor([var['ey'] for var in second_order_stats]) 
        Mxx = torch.tensor([var['Mxx'] for var in second_order_stats])
        Myy = torch.tensor([var['Myy'] for var in second_order_stats]) 
        Mxy = torch.tensor([var['Mxy'] for var in second_order_stats]) 
        n = torch.tensor([var['n'] for var in second_order_stats])

        for i in range(update_steps):
            lr = min_lr + (lr0 - min_lr) * (1 + math.cos(math.pi * i / update_steps)) / 2
            sta = self.agg_second_order_train(Ex, Ey, Mxx, Myy, Mxy, n)
            grad = torch.autograd.grad(sta, self.Pk, retain_graph=False)[0]
            self.Pk.data += lr * grad
            # CosineAnnealingLR
            
    # fed version
    def corr_mix(self, Dx, Dy, f=np.sin, k=20, s=1/6.):
        c_sub = []
        for idd in range(self.k_clients):
            s, k, r, a, b, phi_X, phi_Y = self.cca_rp(Dx[idd], Dy[idd])
            xl = phi_X@a
            yl = phi_Y@b
            cxl = np.column_stack([rankdata(xc, method='ordinal') for xc in xl.reshape(-1,1).T])/float(xl.size)
            cyl = np.column_stack([rankdata(xc, method='ordinal') for xc in yl.reshape(-1,1).T])/float(yl.size)
            c_sub.append(self.cal_second_order(cxl.reshape(-1), cyl.reshape(-1)))
        corr = self.agg_second_order_var(c_sub)
        return corr.numpy()
    