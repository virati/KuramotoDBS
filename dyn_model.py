#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:10:17 2018

@author: virati
Base dynamics class
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand

from DBSpace import nestdict
import networkx as nx


class dyn_model:
    def __init__(self,N=10,R=6,dt=0.01):
        self.state = np.zeros((N*R,1))
        self.L = np.zeros((N*R,N*R))
        
        self.G = nx.from_numpy_matrix(self.L)
        self.state_register = []
        
        self.dt = dt
        self.tvect = np.linspace(0,10,int(1/self.dt))
        
        self.N = N
        self.R = R
        
        self.dynamics = lambda x: 0
    
    def set_connectivity(self,label='structural'):
        pass
    
    def make_L(self):
        inner_connectivity_weight = 5 # default = 5
        inner_connectivity_offset = 1 # default = 0
        R = self.R
        N = self.N
        
        sparsity = 0.98 # default sparsity is 0.99
        for nn in range(R):
            intermed_matrix = inner_connectivity_weight*(inner_connectivity_offset+rand.rand(N,N))
            
            #If we want to binarize
            #intermed_matrix[intermed_matrix < 0.2] = 0
            #intermed_matrix[intermed_matrix >= 0.2] = 1
            
            self.L[nn*N:(nn+1)*N,nn*N:(nn+1)*N] = intermed_matrix
            
            #Now do off-diagonals
            long_mask = rand.randint(100,size=self.L.shape)
            long_mask[long_mask < 100*sparsity] = 0
            long_mask[long_mask >=100*sparsity] = 10
            
            self.L += long_mask
            
                
    def integrator(self):
        k1 = self.dynamics(self.L, self.state) * self.dt
        k2 = self.dynamics(self.L, self.state + .5*k1)*self.dt
        k3 = self.dynamics(self.L, self.state + .5*k2)*self.dt
        k4 = self.dynamics(self.L, self.state + k3)*self.dt
        
        new_state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6
        
        return new_state
    
    def tstep(self):
        #pre step stuff
        
        #step itself through integrator
        new_state = self.integrator()
        
        #Add the state to the register
        self.state = new_state
        self.state_register.append(new_state)
    
    def run(self):
        for tt,time in enumerate(self.tvect):
            self.tstep()
            
    def print_params(self):
        print(self.L)
        
    def plot_state_register(self):
        plt.figure()
        plt.plot(np.array(self.state_register).squeeze())

    def plot_L(self):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(self.L)
        plt.suptitle('Laplacian')
        plt.subplot(1,2,2)
        plt.imshow(self.g_u)
        plt.suptitle('Control')
        
        
class drift_free(dyn_model):
    def dynamics(self):
        return 0

class KNet(dyn_model):
    def __init__(self):
        super(KNet,self).__init__()
        
        self.K = 10
        
        self.make_L()
        self.D = (nx.incidence_matrix(self.G, oriented = True, weight = 'weight')).todense()
        self.state = np.random.normal(0,1,(self.N * self.R))
        
        self.w = rand.normal(0.2,0.1,size=self.state.shape)
        self.dynamics = lambda x,D: self.w - self.K * D * np.sin(D.T * x)


test_net = KNet()
test_net.run()
test_net.plot_state_register()
