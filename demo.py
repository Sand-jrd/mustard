#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Tests of neo-mayo

@author: sand-jrd
"""

from neo_mayo import mayo_estimator
from neo_mayo.utils import ellipse, circle, gaussian
from vip_hci.fits import write_fits


# %% -------------------------------------
# Test mayo estimator constructor

# Choose where to get datas
datadir = "./example-data/"
# datadir = "../PDS70-neomayo/097.C-1001A/K1/"
# datadir = "../Data_challenge/Test_PDS/test_one/"

# Badframes = (0, 35, 36)
# Badframes = list(range(0, 672))
# for k, ii in enumerate(range(0, 672, 5)) : del Badframes[ii-k]

# First step is to build our estimator with the datas
estimator = mayo_estimator(datadir, coro=8, ispsf=False, weighted_rot=True,  Badframes=None)

# Configure your estimator parameters
param = {'w_r'   : 0.05,      # Proportion of Regul over J
        'w_r2'   : 0.05,      # Proportion of Regul2 over J
        'w_way'  : (1, 1),    # You can either work with derotated_cube or rotated cube. Or both
        'gtol'   : 1e-7,      # Gradient tolerence. Stop the estimation when the mean of gradient will hit the value
        'kactiv' : 1,      # Iter before activate regul (i.e when to compute true weight base on w_r proportion)
        'estimI' : True,      # Estimate frames flux is highly recommended !
        'med_sub': False,     # perform a median substraction highly recommended !
        'suffix' : "",        # Name of your simulation (this is optional)
        'res_pos': True,      # Penalize negative residual
        'maxiter': 7}         # Maximum number of iterations (it converge fast tbh)

# %% -------------------------------------
# Configure your regularization if you feel the need

shape = estimator.model.frame_shape
M = gaussian(shape, mu=1, sigma=2)  # You can create a mask with circle, ellipse or gaussian fcts from utils
M = circle(shape, shape[0]//2) + 10*circle(shape, 13)  # You can create a mask with circle, ellipse or gaussian fcts from utils

estimator.configR2(Msk=None, mode="l1", penaliz="X", invert=False)
estimator.configR1(mode="smooth", p_L=1)
#  N.B : or you can juste trust the default parameters and don't call thoses methodes

# %% -------------------------------------
# Test Forward model

if param['estimI'] :  L_est, X_est, flux = estimator.estimate(**param, save=datadir, gif=False, verbose=True)
else : L_est, X_est = estimator.estimate(**param, save=datadir, gif=False, verbose=True)
# N.B : You can stop estimation with esc/ctrl+C. It will terminate properly and results will be return & save

# Complete results are stored in the estimator
res = estimator.res

# The estimator store a lots of stuff :
L0, X0 = estimator.get_initialisation(save="./L0x0/")  # initialization
Lk, Xk, flux_k = estimator.last_iter      # Last iteration
grad = Xk.grad.data[0].detach().numpy()   # Last value of gradient
residual_cube = estimator.get_residual(way="direct")  # Residual cube
residual_cube2 = estimator.get_residual(way="reverse")  # Residual derotated cube

write_fits(datadir + 'residual_cube', estimator.coro.numpy() * residual_cube)
write_fits(datadir + 'residual_cube2', estimator.coro.numpy() * residual_cube2)
