#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:43:02 2018

@author: virati
Main kuramoto class
"""
import numpy as np
import numpy.random as rand
import networkx as nx
import matplotlib.pyplot as plt
import scipy.signal as sig
import pdb

def force_phase(invec):
    return ( invec) % (2 * np.pi )

class KNetwork:
    def __init__(self,R=8,N=80):
        #self.state = np.zeros((R*N,1))
        self.tstep = 0.05 # in seconds
        self.L = np.zeros((R*N,R*N))
        
        self.state = (np.random.uniform(-np.pi,np.pi,size=(R*N,1)))
        
        self.w = np.random.uniform(0.05,0.07,size=(R*N,1))
        
        #self.K = 1/(N*R)
        self.K = 1/45
        self.N = N
        self.R = R
        
        self.make_connectivity()
        
        self.G = nx.from_numpy_matrix(self.L)
        self.D = nx.linalg.incidence_matrix(self.G)
        
        self.r = np.ones((R*N,1))
        
    def make_connectivity(self,form='block'):
        N = self.N
        if form == 'block':
            for nn in range(self.R):
                intermed_matrix = rand.rand(N,N)
                intermed_matrix[intermed_matrix < 0.2] = 0
                intermed_matrix[intermed_matrix >= 0.2] = 1
                
                self.L[nn*N:(nn+1)*N,nn*N:(nn+1)*N] = intermed_matrix
                
                self.L[20,400] = 1
                self.L[320,120] = 1
                self.L[1,600] = 1
                self.L[500,600] = 1
    
            #make random list of "off-diagonal pairs"
            
            #now come up with random pairs and replace those with random numbers
            
    def plot_L(self):
        plt.figure()
        plt.imshow(self.L)
        plt.colorbar()
        
    def dyn(self):
        change = self.K * self.D * np.sin( self.D.T * self.state)
        #pdb.set_trace()
        self.state += self.w - change
        self.state -= np.random.normal(loc=0.0,scale=0.05,size=self.state.shape)
        self.state = force_phase(self.state)
        
        
        self.r += np.random.normal(loc=0.0,scale=0.01,size=self.state.shape)
    
    def run(self,go_time=10.0):
        tvect = np.linspace(0,go_time,go_time/self.tstep)
        self.tvect = tvect
        self.state_matrix = np.zeros((int(go_time/self.tstep),self.N*self.R))
        
        self.state_matrix[0,:] = self.state.squeeze()
        
        for tt,time in enumerate(tvect[:-2]):    
            self.dyn()
            self.state_matrix[tt+1,:] = self.state.squeeze()
            
    def plot_state(self,run=0,region=0):
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.tvect,(self.state_matrix[:,region*self.N:(region+1)*self.N]))
        plt.subplot(2,1,2)
        plt.plot(self.tvect,np.sin(self.state_matrix[:,region*self.N:(region+1)*self.N]))
        plt.suptitle('State')
        
    def measure_region(self,region=0,plotting=True):
        #which region
        phases = self.state_matrix[:,region*self.N:(region+1)*self.N]
        #These are phases
        state = np.exp(1j * phases)
        
        ensemble_state = np.sum(state,axis=1)
        
        if plotting:
            plt.figure()
            plt.plot(np.imag(ensemble_state))
            plt.suptitle('Region Measurement')
            
        return np.imag(ensemble_state)
        
    def measure_psd(self,region=0):
        y_trace = self.measure_region(region=region,plotting=False)
        f,Pxx = sig.welch(y_trace,fs=1/self.tstep)
        
        plt.figure()
        plt.plot(f,10*np.log10(Pxx))
        plt.suptitle('Region PSD')
        
if __name__ == '__main__':
    mainNet = KNetwork()
    #mainNet.plot_L()
    mainNet.run()
    mainNet.plot_state()
    mainNet.measure_region()
    mainNet.measure_psd()
            
