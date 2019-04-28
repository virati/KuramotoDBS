##!/usr/bin/env python3
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
    def __init__(self,N=10,R=6,dt=0.001):
        self.state = np.zeros((N*R,1))
        self.L = np.zeros((N*R,N*R))
        
        self.state_register = []
        
        self.dt = dt
        self.tvect = np.linspace(0,60,int(1/self.dt))
        
        self.N = N
        self.R = R
        
        self.Kt = np.ones_like(self.tvect)
        
        self.dynamics = lambda x: 0
        self.post_dynamics = lambda x: x
    
    def set_connectivity(self,label='structural'):
        self.G = nx.from_numpy_matrix(self.L)
    
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
            
                
    def integrator(self,K):

        k1 = self.dynamics(self.D, self.state,K) * self.dt
        k2 = self.dynamics(self.D, self.state + .5*k1,K)*self.dt
        k3 = self.dynamics(self.D, self.state + .5*k2,K)*self.dt
        k4 = self.dynamics(self.D, self.state + k3,K)*self.dt
        
        new_state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6
        new_state += np.random.normal(0,10,new_state.shape) * self.dt
        
        return new_state
    
    def tstep(self,K=2):
        #pre step stuff
        
        #step itself through integrator
        new_state = self.integrator(K=K)
        new_state = self.post_dynamics(new_state)
        
        #Add the state to the register
        self.state = new_state
        self.state_register.append(new_state)
    
    def run(self):
        for tt,time in enumerate(self.tvect):
            #print('k: ' + str(self.Kt[tt]))
            self.tstep(K=self.Kt[tt])
            
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
        
class dz_model():
    def __init__(self):
        pass
    
    def dz_out(self):
        pass
        
class drift_free(dyn_model):
    def dynamics(self):
        return 0

class KNet(dyn_model):
    def __init__(self,K=2,start_state=None):
        super(KNet,self).__init__(dt=0.0001)
        
        #self.K = K
        #Kt is the timeseries of global connectivity
        self.Kt = np.random.normal(0.1,0.01,size=self.tvect.shape)
        half_pt = int(np.floor(self.Kt.shape[0]/3))
        self.Kt[half_pt:] += K
        
        #self.K is a static parameter used to construct the L
        # Should probably change the names and split these out
        self.K = 6
        
        self.make_L_struct()
        self.set_connectivity()
        
        
        self.D = (nx.incidence_matrix(self.G, oriented = True, weight = 'weight')).todense()
        
        if start_state is None:
            self.state = np.random.normal(0,np.pi,(self.N * self.R,1))
        else:
            self.state = start_state
            
        
        self.w = rand.normal(1,0.1,size=self.state.shape) * 20
        self.dynamics = lambda D,x,k: self.w - k * D * np.sin(D.T * x)
        self.post_dynamics = lambda x: x % (2 * np.pi)
        
        
    def make_L_struct(self,form='block'):
        self.L = np.zeros((self.R*self.N,self.R*self.N))
        N = self.N
        R = self.R
        if form == 'block':
            inner_connectivity_weight = self.K/10 # default = 5
            inner_connectivity_offset = 1 # default = 0
            
            sparsity = 0.98 # default sparsity is 0.99
            for nn in range(self.R):
                intermed_matrix = inner_connectivity_weight*(inner_connectivity_offset+rand.rand(N,N))
                
                #If we want to binarize
                #intermed_matrix[intermed_matrix < 0.2] = 0
                #intermed_matrix[intermed_matrix >= 0.2] = 1
                
                self.L[nn*N:(nn+1)*N,nn*N:(nn+1)*N] = intermed_matrix
                
                #Now do off-diagonals
                long_mask = rand.randint(100,size=self.L.shape)
                long_mask[long_mask < 100*sparsity] = 0
                long_mask[long_mask >=100*sparsity] = self.K/10
                
                self.L += long_mask
                
                #Final thresholding
                #self.L[self.L > 0] = 1
                
        elif form == 'alltoall':
            #self.L = np.ones((R*N,R*N))
            self.L = rand.rand(R*N,R*N)
        
    def plot_state_register(self):
        plt.figure()
        plt.plot(np.sin(np.array(self.state_register).squeeze()))


test_net = KNet(K=4)
test_net.run()
test_net.plot_state_register()
end_state = np.sin(np.array(test_net.state_register[-1]))

#Model2 = KNet(K=0.1,start_state=end_state)
#Model2.run()
#Model2.plot_state_register()