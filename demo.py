#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Tests of neo-mayo

@author: sand-jrd
"""

from neo_mayo import mayo_estimator
from neo_mayo.utils import ellipse, circle, gaussian
import numpy as np
import matplotlib.pyplot as plt
from vip_hci.var import frame_deconvolution
from vip_hci.fits import write_fits

# What you want to do ?
Do_ini = False
gif    = True

# %% -------------------------------------
# Test mayo estimator constructor
# First step is to build our estimator

# Choose where to get datas
# datadir = "./example-data/"
# datadir = "../PDS70-neomayo/097.C-1001A/K1/"
datadir = "../Data_challenge/Test_PDS/test_one/"

# Badframes = (0, 35, 36)
# Badframes = list(range(0, 672))
# for k, ii in enumerate(range(0, 672, 5)) : del Badframes[ii-k]
Badframes = None

# First step is to build our estimator with the datas
coro = 8
estimator = mayo_estimator(datadir, coro=coro, ispsf=False, weighted_rot=True,  Badframes=Badframes)

# Configure your estimator parameters
param = {'w_r'   : 0.06,      # Proportion of Regul over J
        'w_r2'   : 0.04,      # Proportion of Regul2 over J
        'w_way'  : (1, 1),    # You can either work with derotated_cube or rotated cube. Or both
        'gtol'   : 1e-7,      # Gradient tolerence. Stop the estimation when the mean of gradient will hit the value
        'kactiv' : 0,         # Iter before activate regul (i.e when to compute true weight base on w_r proportion)
        'estimI' : True,      # Estimate frames flux is highly recommended !
        'med_sub': True,      # perform a median substraction highly recommended !
        'suffix' : "",        # Name of your simulation (this is optional)
        'maxiter': 5}        # Maximum number of iterations (it converge fast tbh)

# %% -------------------------------------
# Configure an R2 regularization

# Create a mask (you can use circle and ellipse from utils)
shape = estimator.model.frame_shape
#M = ellipse(shape, 85, 50, 13) \
    #- circle(shape, 25)
M = gaussian(shape, mu=1, sigma=2)

# Define your R2 parameters
R2_param = { 'Msk'  : M,
           'mode'   : "mask",
           'penaliz': "X",
           'invert' : False}

# Config R2
estimator.configR2(**R2_param)

# %% -------------------------------------
# First the initialisation

# You can pass agrument for pca as keyargs
pca_arg = {'fwhm': 4, 'asize': 4, 'radius_int': coro}  # Exemple arguments for pca Annular

# Here we compute a pca as initialisation (best and more consistent so far is max_common)
if   Do_ini :  L_ini, X_ini = estimator.initialisation(save=datadir + "/L0X0", Imode="max_common")

# You can also choose do start with L0 = mean(cube), X0 = mean(cube-L0)
elif Do_ini is None :  pass

# I you have already done init and saved it, you can load L0.fits and X0.fits from folder
else :  L_ini, X_ini = estimator.initialisation(from_dir=datadir + "/L0X0")

# %% -------------------------------------
# Test Forward model

if param['estimI'] :  L_est, X_est, flux = estimator.estimate(**param, save=datadir, gif=gif, verbose=True)
else : L_est, X_est = estimator.estimate(**param, save=datadir, gif=gif, verbose=True)

# Complete results are stored in the estimator
res = estimator.res

# The estimator store the last iter as tensor. Hence you can access all sort of things as
# Last value of gradient, last residual per frames, loss evolution...
# everything you need to see if it went as planned
Lk, Xk, flux_k = estimator.last_iter
grad = Xk.grad.data[0].detach().numpy()
reconstructed_cube = estimator.model.forward_ADI(Lk, Xk, flux_k).detach().numpy()
residual_cube = estimator.science_data - reconstructed_cube

write_fits(datadir + 'residual_cube', estimator.coro.numpy() * residual_cube)
write_fits(datadir + 'reconstructed_cube', estimator.coro.numpy() * reconstructed_cube)