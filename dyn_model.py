#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:10:17 2018

@author: virati
Base dynamics class
"""

import numpy as np
import matplotlib.pyplot as plt

class dyn_model:
    def __init__(self,N=10,dt=0.01):
        self.state = np.array((N,1))
        self.L = None
        
        self.state_register = np.zeros_like(self.state)
        
        self.dt = dt
        self.tvect = np.linspace(0,10,int(1/self.dt))
            
    def dynamics(self,in_X):
        return 0
    
    def integrator(self,in_state):
        
        k1 = self.dynamics(in_state) * self.dt
        k2 = self.dynamics(in_state + .5*k1)*self.dt
        k3 = self.dynamics(in_state + .5*k2)*self.dt
        k4 = self.dynamics(in_state + k3)*self.dt
        
        new_state = in_state + (k1 + 2*k2 + 2*k3 + k4)/6
        
        return new_state
    
    def tstep(self):
        new_state = self.integrator(self.state)
        
        self.state_register = np.hstack((self.state_register,new_state))
    
    def run(self):
        for tt,time in enumerate(self.tvect):
            self.tstep()
            
    
    def print_params(self):
        print(self.L)


class KNet(dyn_model):
    def dynamics(self,in_X):
        out_X = - self._L * in_X
        return out_X


test_net = KNet()
test_net.run()
