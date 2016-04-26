import numpy as nm
import networkx as nx

class KMNet(object):
	def __init__(self,N,tstep_end):
		self.G = nx.erdos_renyi_graph(N,0.15)
		self.t = 0
		self.state = nm.zeros(N,tstep_end)

	def init_state(self)
		self.state[:,0] = nm.random.rand(N,1)

	def step(self):
		self.t += 1
		self.state[:,t] = 

def main():
    
	



if __name__=='__main__':
	main()
