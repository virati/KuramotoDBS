#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:31:19 2018

@author: virati (imported and extended from Rohit Konda)
"""
import networkx as nx
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import cmath as c


class KModel:
    def __init__(self, A, K = 1, dt = .05, coh = [1]*8):
        #Graph representation of network
        self.G = nx.from_numpy_matrix(A) #Graph
        self.states = np.matrix([4,1,3,5,6,2]).T #np.matrix(np.random.uniform(0,2*np.pi,size=(6,1)))# memory of phases
        self.w = np.matrix([3.0,3.3,3.6,3.9,4.2,4.5]).T#np.matrix(np.random.normal(3,.2,size=(6,1))) #init intrinsic freq.
        self.K = K #Coupling constant
        self.coh = coh #measures of coherence between each edge
        self.t = 0 #time
        self.dt = dt #time step
    
    #Kuramoto differential equation
    def phase_dev(self,phase):
        D = (nx.incidence_matrix(self.G, oriented = True, weight = 'weight')).todense() #incidence
        N = np.random.normal(0, 10, [len(D[0]), 1])
        return self.w - self.K / len(self.G) * D * -self.edge_func(D.T * self.states[:,-1]) #+N
    
    #euler method approximation of behavior
    def euler_step(self):
        new_state = self.states[:,-1] + self.phase_dev(self.states[:,-1])*self.dt
        new_state = new_state % (2 * np.pi)
        self.t += self.dt
        self.states = np.hstack((self.states,new_state))
       
    #runge-Kutta approximation of behavior
    def runge_kutta_step(self):
        k1 = self.phase_dev(self.states[:,-1])*self.dt
        k2 = self.phase_dev(self.states[:,-1]+ .5*k1)*self.dt
        k3 = self.phase_dev(self.states[:,-1]+ .5*k2)*self.dt
        k4 = self.phase_dev(self.states[:,-1]+ k3)*self.dt
        new_state = self.states[:,-1] + (k1+ 2*k2 + 2*k3 + k4)/6
        new_state = new_state % (2 * np.pi)
        self.t += self.dt
        self.states = np.hstack((self.states,new_state))
    
    #time step function
    def step(self):
        self.euler_step()
        #self.runge_kutta_step()
    
    #coherence function
    def coh_func(self,coh,x):
        c1 = 1 - coh
        c2 = coh
        A = np.sqrt(c1**2+c2**2)
        if coh == 0:
            offset = np.pi/2
        else:
            offset = np.arctan(c1/c2)
        return A*np.sin(x+offset)
    
    #applying function to phase difference for each edge 
    def edge_func(self,M):
        e = [self.coh_func(self.coh[i],M.tolist()[i])[0] for i in range(0,len(self.coh))]
        return np.matrix(e).T   

    def plot_timeseries(self):
        plt.figure()
        plt.plot(np.sin(self.states.T))
        plt.title('State trajectory for all nodes')

def run_model(A, K = 10, t = 10):
    P = KModel(A, K)
    for ts in range(0,int(t/P.dt)):
        P.step()
    return P

def R_sweep(A, Ki = 0, Kf = 10, N = 10):
    def calcR(P):
        Var = []
        for x in np.array(P.states.T):
            z = [c.exp(complex(0,phase)) for phase in x]
            z = sum(z)/len(z)
            Var.append(1-abs(z))
        return sum(Var[round(len(Var)/2):-1])/len(Var)
    KVar = []
    KSpan = np.linspace(Ki,Kf, num = N)
    for K in KSpan:
        P = run_model(A,K,t = 10)
        KVar.append(calcR(P))
    plt.figure()
    plt.plot(KSpan,KVar)
    plt.xlabel('K values')
    plt.ylabel('Mean Variance over 10s')
    plt.title('Global Variance vs K')
    plt.show()
    return (KSpan, KVar)

def animate(K,t):
    import time
    import pylab as plt
    t_full = t
    radii = np.ones(K.states[:,-1].shape)
    
    fig,ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
    ax.set_rmax(10)
    ax.set_title('Phase Propogation')
    for i in range(0,int(t_full/K.dt)):
        K.step()
        ax.lines = []
        ax.stem(K.states[:,-1],radii)
        fig.canvas.draw()
        if(K.dt > .06):
            time.sleep(K.dt-.06)
            
if __name__ == '__main__':
    nodes = [1,2,3,4,5,6]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([(1,2,1),(2,3,1),(1,6,1),(1,3,1),(3,4,1),(4,5,1),(4,6,1),(5,6,1)])
    
    
    #Quick plot of G
    nx.draw(G)
    A = nx.to_numpy_matrix(G)
    print(A)
    
    
    K = KModel(A)
    #animate(K,10)
    model_out = run_model(A,t=10,K=10)
    model_out.plot_timeseries()