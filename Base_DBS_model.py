# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:42:23 2016

@author: virati, rohit
Basefile for Kuramoto style modeling
"""

import networkx as nx
import numpy as nm

import scipy.signal as sig
import matplotlib.pyplot as plt

class KModel:
    G = nx.Graph()
    phase = nm.array([]) 
    states = nm.array([])
    timestep = 0    
    
    def __init__(freq_band = 0):
        self.G.add_nodes_from([1,2,3,4,5,6])
        self.G.add_edges_from([(1,2),(2,3),(1,3),(3,4),(4,5),(4,6),(5,6)])
        
        #if states are oscillatory
        self.phase = nm.random.uniform(-nm.pi/2,nm.pi/2,[nodes,1])
        self.intrinsic_freq = nm.random.normal(0.5,0.1,[nodes,1])
        self.states = self.int_func(self.phase)     
        
    def give_graph(self):
        return self.G
        
    def give_states(self):
        return self.phase
        
    def state_blip(self,node):
        self.states[node] = -nm.pi/2
        
    def step_time(self):
        self.timestep += 1
        self.propagate_state()

    def int_func(self,func,phase):
        if func == '0':
            return nm.array([0] * len(phase))
        elif func == 'self':
            return phase
        elif func == 'sin':
            return nm.sin(phase)
        elif func == 'cos':
            return nm.cos(phase)
        else:
            raise ValueError('not appropriate interaction func')
        
        
    def propagate_state(self):
        self.phase -= self.intrinsic_freq + 1/6* nx.laplacian_matrix(self.G) * self.int_func(self.phase)
        self.phase = (self.phase + nm.pi) % (2 * nm.pi) - nm.pi        
        
        self.states = nm.hstack((self.states,self.int_func(self.phase)))

    def give_state_course(self):
        return self.states
    
def main():
    UC = KModel()

    for ts in range(1,1000):
        UC.step_time()
    
    plt.figure()
    nx.draw(UC.give_graph())
    
    plt.figure()
    plt.plot(UC.give_states())

    plt.figure()
    plt.plot(UC.give_state_course().T)   
    plt.xlim((0,10))     
    
    plt.show()
    
    print(UC.give_states())
    
    
if __name__=='__main__':
    main()