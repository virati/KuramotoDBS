#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:34:07 2020

@author: virati
This file will do a basic Jansen Rit with autoDyn
"""

import numpy as np
import auto

def sig(x):
    return 1/(1 + gamma * np.exp(-x))

def JR_unit(params,x,I):
    alpha = params['alpha']
    beta = params['beta']
    

def JR_net(params,x):
    L = params['L']
    
    for ii in range(x):
        