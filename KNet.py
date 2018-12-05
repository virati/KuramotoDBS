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

import pdb

def ew_multi(*args):
    amatr = np.ones_like(args[0])
    for arg in args:
        amatr = np.multiply(amatr,arg)
        
    return amatr
        
#%%  

class KNet:
    def __init__(self, K = 10, dt =.01,R=6,N=1):
        
        self.make_connectivity(R,N,form='block')
        self.make_control(R,N,weight=1)
        self.make_input_G(R,N,weight=2)
        
        self.N = N
        self.R = R
        
        self.G = nx.from_numpy_matrix(self.L) #Graph
        #self.states = np.matrix([4,1,3,5,6,2]).T ## memory of phases
        #Phase states
        self.states = np.matrix(np.random.uniform(0,2*np.pi,size=(R*N,1)))
        
        #Radius States
        self.r_states = np.matrix(np.random.uniform(0.1,5,size=(R*N,1)))
        self.r_centers = np.random.uniform(0.1,19,size=(R*N,1))
        self.r_bound = 20
        
        #self.w = np.matrix([3.0,3.3,3.6,3.9,4.2,4.5]).T#np.matrix(np.random.normal(3,.2,size=(6,1))) #init intrinsic freq.
        self.w = np.matrix(np.random.normal(20,0.5,size=(R*N,1)))
        self.t = 0 #time
        self.K = K #coupling constant
        self.dt = dt #time step
        self.step_num = 0
        
        self.o_stat = []
        
        self.K_i = 10
        
    def draw_network(self):
        plt.figure()
        nx.draw(self.G)

    def plot_connectivity(self):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(self.L)
        plt.suptitle('Laplacian')
        plt.subplot(1,2,2)
        plt.imshow(self.g_u)
        plt.suptitle('Control')
        
    '''
    Weight defaults to 2
    '''
    
    def make_control(self,R,N,weight=2):
        self.g_u = np.zeros((R*N,R*N))
        #Off diagonal strong connectivities for the cos factor
        ctrl_matrix = weight*np.ones((N,N))
        
        edge_couples = [[2,3],[0,5],[1,2]]
        
        for edge in edge_couples:
            r_1 = edge[0]
            r_2 = edge[1]
            self.g_u[r_1*N:(r_1+1)*N,r_2*N:(r_2+1)*N] = ctrl_matrix
            #do the other direction as well
            self.g_u[r_2*N:(r_2+1)*N,r_1*N:(r_1+1)*N] = ctrl_matrix
        
        
        self.G_ctrl = nx.from_numpy_matrix(self.g_u)
    
    def make_input_G(self,R,N,weight=2):
        self.g_i = np.zeros((R*N,R*N))
        #Off diagonal strong connectivities for the cos factor
        ctrl_matrix = weight*np.ones((N,N))
        
        edge_couples = [[1,4],[1,5],[1,2]]
        
        for edge in edge_couples:
            r_1 = edge[0]
            r_2 = edge[1]
            self.g_i[r_1*N:(r_1+1)*N,r_2*N:(r_2+1)*N] = ctrl_matrix
            #do the other direction as well
            self.g_i[r_2*N:(r_2+1)*N,r_1*N:(r_1+1)*N] = ctrl_matrix
        
        
        self.G_inp = nx.from_numpy_matrix(self.g_i)
        
    def make_connectivity(self,R=6,N=1,form='block'):
        self.L = np.zeros((R*N,R*N))
        if form == 'block':
            inner_connectivity_weight = 5 # default = 5
            inner_connectivity_offset = 1 # default = 0
            
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
                
                #Final thresholding
                #self.L[self.L > 0] = 1
                
        elif form == 'alltoall':
            #self.L = np.ones((R*N,R*N))
            self.L = rand.rand(R*N,R*N)
            
        elif form == 'UCircuit':
            nodes = [1,2,3,4,5,6]
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_weighted_edges_from([(1,2,1),(2,3,1),(1,6,1),(1,3,1),(3,4,1),(4,5,1),(4,6,1),(5,6,1)])
            self.L = nx.to_numpy_matrix(G)
    
    
    #Kuramoto differential equation
    def phase_dev(self,phase):
        D = (nx.incidence_matrix(self.G, oriented = True, weight = 'weight')).todense() #incidence
        N = np.random.normal(0, 20, [len(D[0]), 1])
        
        # How to handle Rs
        #rsq = self.r_states[:,-1] * self.r_states[:,-1].T
        
        #How to bring phases together; this is the intrinsic alpha mode
        bring_in = self.w - np.multiply(self.r_states[:,-1],self.K / len(self.G) * D * np.sin(D.T * self.states[:,-1])) + N
        
        # Control is done HERE
        D_ctrl = (nx.incidence_matrix(self.G_ctrl, oriented = True, weight = 'weight')).todense()
        #bring_out = self.K_u / len(self.G) * D_ctrl * np.cos(D_ctrl.T * self.states[:,-1])
        bring_stim = self.K_u / len(self.G) * D_ctrl * np.sin(D_ctrl.T * self.states[:,-1]) + D_ctrl * 200 * D_ctrl.T * np.ones_like(self.states[:,-1])
        
        # Inputs are done HERE
        # TREAT STIM LIKE A "PATHOLOGY FIXER" to let the brain's intrinsic dynamics do their thing.
        # So, basically, INPUTS should be desyncing and STIM should be pure *blocking*
        D_inp = (nx.incidence_matrix(self.G_inp, oriented = True, weight = 'weight')).todense()
        input_out = self.K_i / len(self.G) * D_inp * np.cos(D_inp.T * self.states[:,-1])
        
        
        return bring_in + bring_stim + input_out + N
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
    
    def r_dyn(self,rin):
        #we want to take the r inputs and subtract the region's phasor amplitude
        #state are the PHASES
        #pdb.set_trace()
        phasors = np.multiply(rin,np.exp(1j * self.states[:,-1]))
        r_change = np.zeros_like(self.states[:,-1])
        N = np.random.normal(0, 2, self.states[:,-1].shape)
        
        region_phasor = np.zeros((self.R,1),dtype=complex)
        
        for rr in range(self.R):
            region_phasor[rr] = 1/(self.N) * np.sum(phasors[rr*self.N:(rr+1)*self.N])
            r_change[rr*self.N : (rr+1) * self.N] = np.abs(region_phasor[rr])
            
        r_change = - (rin - 2) - 5*np.multiply(r_change,r_change) + N
        
        return r_change
    
    def POLYr_dyn(self,rin,form='quintic'):
        
        if form == 'cubic':
            N = np.random.normal(0, 10, (rin.shape[0], 1))
            return np.tanh(np.multiply(np.multiply((rin - self.r_centers),(rin - self.r_bound)),(rin+self.r_bound)) + N)
        
        elif form == 'quintic':
            N = np.random.normal(0,50, (rin.shape[0], 1))
            return 5 * np.tanh((ew_multi((rin - self.r_centers),(rin - self.r_centers - 5),(rin - self.r_centers + 5),(rin - self.r_bound),(rin+self.r_bound)))/5)  + N
    
    def plot_r_stats(self):
        #plt.figure()
        #plt.plot(np.std(self.states,axis=0).T)
        plt.plot(self.r_states.T)
        
    #Runge Kutta
    def step_RK(self,K_u=0):
        self.K_u = K_u
        
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
        new_r_state = np.maximum(self.r_states[:,-1] + ((p1 + 2*p2 + 2*p3 + p4)/6),0)
        self.r_states = np.hstack((self.r_states,new_r_state))
        
        #%%
        #order stat handling
        self.o_stat.append((self.order_stat()))
        
        self.t += self.dt
        self.step_num +=1
        
    def order_stat(self):
        #compute the order statistic for each node
        aggr_r = []
        for rr in range(self.R):
            aggr_r.append(np.sum(np.exp(1j * self.states[self.N * rr: self.N * (rr+1),-1])))
            
        return aggr_r
        
        

    def plot_timecourse(self):
        
        tvect = np.linspace(0,(self.step_num+1) * self.dt,self.step_num+1)
        
        plt.figure()
        
        plt.subplot(311)
        plt.plot(tvect,np.sin(self.states.T))
        plt.xlabel('Time Steps')
        plt.ylabel('x')
        plt.title('t')
        
        plt.subplot(312)
        #plt.plot(tvect,(self.r_states.T))
        #plt.plot(np.abs(self.o_stat))
        plt.plot(np.abs(self.o_stat))
        #plt.hlines(self.r_centers,xmin=0,xmax=10)
        plt.title('Order statistics')
    
        
        plt.subplot(313)
        aggr_sums = np.array(self.o_stat) #/ np.sum(np.array(self.o_stat),axis=0,keepdims=True)
        aggr = np.sum(aggr_sums,axis=1)
        plt.plot(np.abs(aggr))
        plt.title('Across Rs')
        
        plt.figure()
        plt.plot(self.r_states.T)
#%%       
def run_model(K = 10, t = 5):
    P = KNet(K,R=6,N=5,dt=0.001)
    for ts in range(0,int(t/P.dt)):
        if ts > t/(2*P.dt):
            K_u = 100
        else:
            K_u = 0
            
        P.step_RK(K_u =K_u)
    return P


if __name__=='__main__':
    modelOut = run_model(K=5)
    modelOut.plot_timecourse()
    modelOut.draw_network()
    modelOut.plot_connectivity()