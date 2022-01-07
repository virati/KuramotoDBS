#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:46:38 2020

@author: virati
Alpha-Beta push-pull modeling
Given what I saw in the SCCwm-DBS recordings, with SCCwm-DBS immediately effecting $\alpha$ changes while long-term changes in delta+beta reflected depression changes, this model is my attempt to see what could link the two in a way that is parsimonious with other (unseen) structure
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

t = np.linspace(0,10,10000)
u = np.zeros_like(t)

def dropout_squares(f_c):
    u[-5000:] = 1
    x = np.sin(2 * np.pi * f_c * t) + (1-u)*(1/3 * np.sin(2 * np.pi * f_c * 3 * t) + 1/5*np.sin(2 * np.pi * f_c * 5 * t) + 1/7*np.sin(2 * np.pi * f_c * 7 * t))
    
    plt.figure()
    plt.plot(t,x)
    plt.figure()
    F,T,SG = sig.spectrogram(x,fs=100)
    plt.pcolormesh(T,F,np.abs(SG))

dropout_squares(f_c=4)