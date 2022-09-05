#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Tests of neo-mayo

@author: sand-jrd
"""

from mustard import mustard_estimator
from vip_hci.fits import open_fits

# Stuff to help you build a custom R2
from mustard.utils import ellipse, circle, gaussian, unpack_science_datadir


# %% -------------------------------------
# Load your data and build the estimator.

# Load the ADI cube and the associated angles
datadir = "./example-data/"
datadir = "../PDS70-neomayo/095.C-0298A/H3/"
#datadir = "../DISK/MWC758/"

science_data = open_fits(datadir+"cube.fits")
angles = open_fits(datadir+"angles.fits")

# (obsolete) There is a tool that gets data in from your repository.
# datadir = "../PDS70-neomayo/1100.C-0481D/K2/"
# science_data, angles, psf =  unpack_science_datadir(datadir)

# Then build your  estimator with the datas
estimator = mustard_estimator(science_data, angles, coro=8, pupil="edge", Badframes=range(27, 41))

# Configure your estimator parameters
param = {'w_r'   : 0.05,      # Proportion of Regul over J
        'w_r2'   : None,      # Proportion of Regul2 over J
        'w_way'  : (1, 0),    # You can either work with derotated_cube or rotated cube. Or both
        'gtol'   : 1e-7,      # Gradient tolerence. Stop the estimation when the mean of gradient will hit the value
        'kactiv' : 0,         # Iter before activate regul (i.e when to compute true weight base on w_r proportion)
        'estimI' : "L",    # Estimate frames flux is highly recommended ! possible value : {"Frame","L","Both"}
        'med_sub': False,      # perform a median substraction highly recommended !
        'weighted_rot' : True,# Compute weight for each frame according to PA angle separations.
        'suffix' : "hfm",        # Name of your simulation (this is optional)
        'init_maxL': False,
        'res_pos': False,     # Penalize negative residual
        'maxiter': 30}        # Maximum number of iterations (it converge fast tbh)

# %% -------------------------------------
# Configure your regularization if you feel the need

shape = estimator.model.frame_shape
M = gaussian(shape, mu=1, sigma=2)  # You can create a mask with circle, ellipse or gaussian fcts from utils
M = circle(shape, 13)  # You can create a mask with circle, ellipse or gaussian fcts from utils

estimator.configR2(Msk=None, mode="l1", penaliz="L", invert=False)
estimator.configR1(mode="smooth", p_L=2)
#  N.B : or you can juste trust the default parameters and don't call thoses methodes

# %% -------------------------------------
# Test Forward model

# In case you didn't provid a savedir before and you don't want it to be print at workingdir
estimator.set_savedir(datadir+param['suffix'])

estimator.estimate_halos(full_output=True, save=True)

if False :
        if param['estimI'] :  L_est, X_est, flux = estimator.estimate(**param, save=datadir, gif=False, verbose=True)
        else : L_est, X_est = estimator.estimate(**param, save=datadir, gif=False, verbose=True)
# N.B : You can stop estimation with esc/ctrl+C. It will terminate properly and results will be return & save


# Complete results are stored in the estimator ...
res = estimator.res

# You can access easily to what you need with get methods  :
L0, X0 = estimator.get_initialisation(save=True)  # initialization

estimator.get_residual(way="direct", save=True)  # Residual cube
estimator.get_reconstruction(way="direct", save=True)  # Reconstruction
estimator.get_cube_no_halo(save=True)

# estimator.get_evo_convergence(show=True, save=True)
# estimator.get_flux(show=True, save=True)
# estimator.get_rot_weight(show=True, save=True)

estimator.mustard_results()
