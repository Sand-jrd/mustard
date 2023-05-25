#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Tests of neo-mayo

@author: sand-jrd
"""

# Import Mustard estimator
from mustard import mustard_estimator

# Also, this will help you build the mask fo regularization
from mustard.utils import gaussian

# Misc
from vip_hci.fits import open_fits
from vip_hci.preproc import cube_crop_frames

# %% -------------------------------------
# Load your data and build the estimator.

# Load the ADI cube (1 channel) and the associated angles
datadir = "./example-data/"

science_data = open_fits(datadir+"cube")
angles = open_fits(datadir+"angles")[0]

# Note : Don't hestiate to crop the cubes it is time/ressources-consuming
science_data = cube_crop_frames(science_data, 256)

# %% -------------------------------------
# Then build your  estimator with the datas

estimator = mustard_estimator(science_data, angles=angles, coro=10,  pupil="edge", Badframes=None, hid_mask=None)

# And configure your estimator parameters
param = {'w_r'   : 0.15,         # Proportion of Regul over J
        'w_r2'   : 0.03,         # Proportion of Regul2 over J
        'w_r3'   : 0.001,        # Proportion of Regul2 over J
        'w_way'  : (1, 0),       # You‡ can either work with derotated_cube or rotated cube. Or both
        'gtol'   : 1e-100,       # Gradient tolerence. Stop the estimation when the mean of gradient will hit the value
        'kactiv' : 0,            # Iter before activate regul (i.e when to compute true weight base on w_r proportion)
        'estimI' : "None",       # Estimate frames flux is highly recommended ! possible value : {"Frame","L","Both"}
        'weighted_rot' : False,  # Compute weight for each frame according to PA angle separations.
        'suffix' : "test",       # # Name of your simulation (this is optional)
        'maxiter': 13,
        'mask_L': (0 , 100),
        'init_maxL': True}


# %% -------------------------------------

# Configure your regularization if you feel the need
# or you can juste try the default parameters

shape = estimator.model.frame_shape
M = gaussian(shape, mu=0, sigma=0.7) + gaussian(shape, mu=0, sigma=0.2) + 1*gaussian(shape, mu=0, sigma=0.05)

# R1 : Smooth regualization
estimator.configR1(mode="smooth", p_L=3, p_X=2)

# R2 : Mask regualrization
estimator.configR2(Msk=M, mode="mask", penaliz="Both", invert=True)

# R3 : Extra regul (either smooth or l1 on L and/or X)
estimator.configR3(mode="l1", p_L=0, p_X=0.1)



# %% -------------------------------------

## You are now ready to start the estimation

L_est, X_est = estimator.estimate(**param, save=datadir, gif=False, verbose=True)
# N.B : You can stop estimation with esc/ctrl+C. It will terminate properly and results will be return & save

# Complete results are stored in the estimator ...
res = estimator.res

# You can access easily to what you need with the gets methods  :
L0, X0 = estimator.get_initialisation(save="./L0x0/")  # initialization
residu = estimator.get_residual(way="direct", save=True)        # Residual cube
reconstruction = estimator.get_reconstruction(way="direct", save=True)  # Reconstruction
