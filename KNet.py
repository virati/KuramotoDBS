#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 18:25:20 2018

@author: virati, Rohit Konda
PO Model version from Rohit
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand

def ew_multi(*args):
    amatr = np.ones_like(args[0])
    for arg in args:
        amatr = np.multiply(amatr,arg)
        
    return amatr
        
#%%  

class KNet:
    def __init__(self, K = 10, dt =.01,R=6,N=1):
        
        self.L = np.zeros((R*N,R*N))
        self.make_connectivity(R,N,form='block')
        
        self.G = nx.from_numpy_matrix(self.L) #Graph
        #self.states = np.matrix([4,1,3,5,6,2]).T ## memory of phases
        #Phase states
        self.states = np.matrix(np.random.uniform(0,2*np.pi,size=(R*N,1)))
        
        #Radius States
        self.r_states = np.matrix(np.random.uniform(0.1,5,size=(R*N,1)))
        self.r_centers = np.random.uniform(0.1,19,size=(R*N,1))
        self.r_bound = 20
        
        #self.w = np.matrix([3.0,3.3,3.6,3.9,4.2,4.5]).T#np.matrix(np.random.normal(3,.2,size=(6,1))) #init intrinsic freq.
        self.w = np.matrix(np.random.normal(20,0.2,size=(R*N,1)))
        self.t = 0 #time
        self.K = K #coupling constant
        self.dt = dt #time step
        self.step_num = 0
        
        
    def draw_network(self):
        plt.figure()
        nx.draw(self.G)

    def plot_connectivity(self):
        plt.figure()
        plt.imshow(self.L)
        plt.suptitle('Laplacian')
        
    def make_connectivity(self,R=6,N=1,form='block'):
        if form == 'block':
            sparsity = 0.99
            for nn in range(R):
                intermed_matrix = rand.rand(N,N)/3
                
                #If we want to binarize
                #intermed_matrix[intermed_matrix < 0.2] = 0
                #intermed_matrix[intermed_matrix >= 0.2] = 1
                
                self.L[nn*N:(nn+1)*N,nn*N:(nn+1)*N] = intermed_matrix
                
                #Now do off-diagonals
                long_mask = rand.randint(100,size=self.L.shape)
                long_mask[long_mask < 100*sparsity] = 0
                long_mask[long_mask >=100*sparsity] = 3
                
                self.L += long_mask
                
                #Final thresholding
                #self.L[self.L > 0] = 1
                
        elif form == 'alltoall':
            self.L = np.ones((R*N,R*N))
            
        elif form == 'UCircuit':
            nodes = [1,2,3,4,5,6]
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_weighted_edges_from([(1,2,1),(2,3,1),(1,6,1),(1,3,1),(3,4,1),(4,5,1),(4,6,1),(5,6,1)])
            self.L = nx.to_numpy_matrix(G)
    
    
    #Kuramoto differential equation
    def phase_dev(self,phase):
        D = (nx.incidence_matrix(self.G, oriented = True, weight = 'weight')).todense() #incidence
        N = np.random.normal(0, 10, [len(D[0]), 1])
        
        # How to handle Rs
        rsq = self.r_states[:,-1] * self.r_states[:,-1].T
        
        return self.w - self.K / len(self.G) * rsq/(10**2) * D * np.sin(D.T * self.states[:,-1]) + N
        #return self.w - np.multiply((1/self.r_states[:,-1]),self.K / len(self.G) * D * np.sin(D.T * self.states[:,-1]) + N)
    
    # 4th order Runge-Kutta approximation
    def DEPRstep(self):
        new_state = self.states[:,-1] + self.phase_dev(self.states[:,-1])*self.dt
        new_state = new_state % (2 * np.pi)
        self.states = np.hstack((self.states,new_state))
        
        #Work on r now
        r_state = self.r_states[:,-1]
        r_state += (r_state - self.r_centers) * (r_state - self.r_bound) * (r_state + self.r_bound)
        self.r_states = np.hstack((self.r_states,r_state))
        
        #Track and update states
        self.t += self.dt
        
        self.step_num += 1
    
    def r_dyn(self,rin,form='quintic'):
        
        if form == 'cubic':
            N = np.random.normal(0, 10, (rin.shape[0], 1))
            return np.tanh(np.multiply(np.multiply((rin - self.r_centers),(rin - self.r_bound)),(rin+self.r_bound)) + N)
        
        elif form == 'quintic':
            N = np.random.normal(0,50, (rin.shape[0], 1))
            return 5 * np.tanh((ew_multi((rin - self.r_centers),(rin - self.r_centers - 5),(rin - self.r_centers + 5),(rin - self.r_bound),(rin+self.r_bound)))/5)  + N
    
    def plot_r_stats(self):
        plt.figure()
        plt.plot(np.std(self.states,axis=0).T)
        
    #Runge Kutta
    def step_RK(self):
        k1 = self.phase_dev(self.states[:,-1])*self.dt
        k2 = self.phase_dev(self.states[:,-1]+ .5*k1)*self.dt
        k3 = self.phase_dev(self.states[:,-1]+ .5*k2)*self.dt
        k4 = self.phase_dev(self.states[:,-1]+ k3)*self.dt
        new_state = self.states[:,-1] + (k1+ 2*k2 + 2*k3 + k4)/6
        new_state = new_state % (2 * np.pi)
        self.states = np.hstack((self.states,new_state))  
        
        #Work on the Rs now
        p1 = self.r_dyn(self.r_states[:,-1])*self.dt
        p2 = self.r_dyn(self.r_states[:,-1] + 0.5*p1)*self.dt
        p3 = self.r_dyn(self.r_states[:,-1] + 0.5*p2)*self.dt
        p4 = self.r_dyn(self.r_states[:,-1] + p3)*self.dt
        new_r_state = self.r_states[:,-1] + ((p1 + 2*p2 + 2*p3 + p4)/6)
        self.r_states = np.hstack((self.r_states,new_r_state))
        
        self.t += self.dt
        self.step_num +=1

    def plot_timecourse(self):
        
        tvect = np.linspace(0,(self.step_num+1) * self.dt,self.step_num+1)
        plt.figure()
        plt.subplot(211)
        plt.plot(tvect,np.sin(self.states.T))
        plt.xlabel('Time Steps')
        plt.ylabel('x')
        plt.title('t')
        
        plt.subplot(212)
        plt.plot(tvect,(self.r_states.T))
        plt.hlines(self.r_centers,xmin=0,xmax=10)
#%%       
def run_model(K = 10, t = 10):
    P = KNet(K,N=5)
    for ts in range(0,int(t/P.dt)):
        P.step_RK()
    return P


if __name__=='__main__':
            
    
    modelOut = run_model()
    modelOut.plot_timecourse()
    modelOut.draw_network()
    modelOut.plot_connectivity()