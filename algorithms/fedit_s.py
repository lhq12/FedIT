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

class FedIT_CS_S(Base):
    """
    Federated Independence Test with Copular Alignment and Stacking Aggetation

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
            "sum": aggregate all the sample
        
        gamma approximate. Default: True.
        """
        super(FedIT_CS_S, self).__init__(k_clients, n_fed, x_fed, y_fed, alpha=alpha)
        
        self.select_method = select_method
        
        if self.select_method not in ["opt", "sum"]:
            raise NotImplementedError()
        
        if self.select_method == "opt": # need to split the sample
            self.fed_train_x = [x[:int(len(x)*split_ratio)] for x in self.fed_npdata_x]
            self.fed_train_y = [y[:int(len(y)*split_ratio)] for y in self.fed_npdata_y]
            self.fed_train_n = np.array([int(len(x)*split_ratio) for x in self.fed_npdata_x])
            self.fed_test_x = [x[int(len(x)*split_ratio):] for x in self.fed_npdata_x]
            self.fed_test_y = [y[int(len(y)*split_ratio):] for y in self.fed_npdata_y]
            
    def perform_test(self, B=100):
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
        
        sub_id = self.aggstep()
        
        if self.select_method == "opt":
            DX = self.fed_test_x
            DY = self.fed_test_y
        elif self.select_method == "sum":
            DX = self.fed_npdata_x
            DY = self.fed_npdata_y
        sta = self.corr_mix(sub_id, DX, DY)
        sta_list = []
        for _ in range(B):
            DX_perm = [np.random.permutation(x) for x in DX]
            sta_list.append(self.corr_mix(sub_id, DX_perm, DY))
        
        self.sta_list = sta_list
        p_value = (min((sum(sta>=sta_list))/len(sta_list), (sum(sta<sta_list)+1)/len(sta_list))) ## two side test
        
        h0_rejected = (p_value < self.alpha/2)
        
        return {
            "alpha": self.alpha,
            "p_value": p_value,
            "sub_id": sub_id,
            "h0_rejected": h0_rejected
        }
    
    def aggstep(self):
        if self.select_method == "opt":
            z_stats = []
            subsets = self.all_nonempty_subsets(self.k_clients)
            for subset in subsets:
                cxl = np.concatenate([self.client_status[i][0] for i in subset])
                cyl = np.concatenate([self.client_status[i][1] for i in subset])
                sta = np.corrcoef(cxl.reshape(-1), cyl.reshape(-1))[0,1]
                sta_list = []
                B = 100
                for _ in range(B):
                    DX_perm = [np.random.permutation(x) for x in self.fed_train_x]
                    sta_list.append(self.corr_mix(subset, DX_perm, self.fed_train_y))

                z_stat_sub = 1 - (min((sum(sta>=sta_list))/len(sta_list), (sum(sta<sta_list)+1)/len(sta_list))) ## two side test
                z_stats.append(z_stat_sub)
            sub_id = subsets[np.argmax(z_stats)]
        elif self.select_method == "sum":
            sub_id = [i for i in range(self.k_clients)]
        return sub_id
    
    ### corr with mix of subsets
    def corr_mix(self, sub_id, Dx, Dy, f=np.sin, k=20, s=1/6.):
        c_sub = []
        for idd in sub_id:
            s, k, r, a, b, phi_X, phi_Y = self.cca_rp(Dx[idd], Dy[idd])
            xl = phi_X@a
            yl = phi_Y@b
            cxl = np.column_stack([rankdata(xc, method='ordinal') for xc in xl.reshape(-1,1).T])/float(xl.size)
            cyl = np.column_stack([rankdata(xc, method='ordinal') for xc in yl.reshape(-1,1).T])/float(yl.size)
            c_sub.append((cxl, cyl))

        cx_all = np.concatenate([x[0] for x in c_sub])
        cy_all = np.concatenate([x[1] for x in c_sub])

        corr = np.corrcoef(cx_all.reshape(-1), cy_all.reshape(-1))[0,1]
        return corr
    
    