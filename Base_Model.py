# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:42:23 2016

@author: virati
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
    
    def __init__(self,nodes=6,edge_factor=0.2):
        self.G = nx.gnm_random_graph(nodes,nodes*nodes * edge_factor)
        
        #if states are oscillatory
        self.phase = nm.random.uniform(-nm.pi/2,nm.pi/2,[nodes,1])
        self.intrinsic_freq = nm.random.normal(0.5,0.1,[nodes,1])
        self.states = nm.sin(self.phase)        
        
    def give_graph(self):
        return self.G
        
    def give_states(self):
        return self.phase
        
    def state_blip(self,node):
        self.states[node] = -nm.pi/2
        
    def step_time(self):
        self.timestep += 1
        self.propagate_state()
        
        
    def propagate_state(self):
        self.phase -= self.intrinsic_freq + 1/6* nx.laplacian_matrix(self.G) * nm.sin(self.phase)
        self.phase = ( self.phase + nm.pi ) % (2 * nm.pi) - nm.pi        
        
        self.states = nm.hstack((self.states,nm.sin(self.phase)))

    def give_state_course(self):
        return self.states
    
def main():
    n_nodes = 6
    UC = KModel(n_nodes,0.2)
    t = range(1,1000)
    #state_raster = numpy.zeros([n_nodes,1])

    for ts in t:
        #if ts % 10 == 0:
        #    UC.state_blip(3)
        UC.step_time()
        #state_raster = nm.hstack((state_raster,UC.give_states()))
    
    plt.figure()
    nx.draw(UC.give_graph())
    
    plt.figure()
    plt.plot(UC.give_states())

    plt.figure()
    plt.plot(UC.give_state_course().T)   
    plt.xlim((0,5))     
    
    plt.show()
    
    print(UC.give_states())
    
    
if __name__=='__main__':
    main()