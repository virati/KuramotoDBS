'''
Author: Vineet Tiruvadi

This script will prototype the link between a network of Kuramoto oscillations and behavioral trajectories

'''
import autodyn


KModel = ady.kuramoto.standard_model()
BModel = ady.behavior.tabula_raza()

sim = ady.sim.Simulator(KModel, BModel)

sim.run(T=1000, dt=0.01)

sim.coverage(('measurement', 'behavior'))
