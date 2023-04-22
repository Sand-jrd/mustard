#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Tests of neo-mayo

@author: sand-jrd
"""

from mustard import mustard_estimator
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import cube_crop_frames, frame_crop

# Stuff to help you build a custom R2
from mustard.utils import circle, gaussian, radial_profil, ellipse

# %% -------------------------------------
# Load your data and build the estimator.

# Load the ADI cube and the associated angles
datadir = "../DISK/HIP_73145/"
datadir = "../dataset_rexpaco/"
#datadir = "./example-data/"
science_data = open_fits(datadir+"cube")
#science_data = cube_crop_frames(open_fits(datadir+"cube"),199, force=True)
import numpy as np
#science_data = 1/np.max(science_data, axis=(1,2), keepdims=True)* science_data
angles = -open_fits(datadir+"angles")[0] # + 135.99 - 1.75
scales = None

#import numpy as np
#science_data /= np.max(science_data, axis=0)
#write_fits(datadir+"cube_chan1", science_data)

#bframe = np.array([0,1,2,3,4,5,6,7,8,9,10,11,44,45,48,46,47,49])

# Then build your  estimator with the datas
estimator = mustard_estimator(science_data, angles=angles, scale=scales, coro=10,  pupil="edge", Badframes=None, hid_mask=None)

# Configure your estimator parameters
param = {'w_r'   : 0.15,          # Proportion of Regul over J
        'w_r2'   : 0.03,         # Proportion of Regul2 over J
        'w_r3'   : 0.001,         # Proportion of Regul2 over J
        'w_way'  : (1, 0),       # You‡ can either work with derotated_cube or rotated cube. Or both
        'gtol'   : 1e-100,        # Gradient tolerence. Stop the estimation when the mean of gradient will hit the value
        'kactiv' : 0,            # Iter before activate regul (i.e when to compute true weight base on w_r proportion)
        'estimI' : "None",       # Estimate frames flux is highly recommended ! possible value : {"Frame","L","Both"}
        'weighted_rot' : False,  # Compute weight for each frame according to PA angle separations.
        'suffix' : "test-ml",      # "ASDI",     # Name of your simulation (this is optional)
        'maxiter': 13,
        'mask_L': (0 , 100),
        'init_maxL': True}


# %% -------------------------------------
# Configure your regularization if you feel the need

shape = estimator.model.frame_shape
M = gaussian(shape, mu=0, sigma=0.7) + gaussian(shape, mu=0, sigma=0.2) + 1*gaussian(shape, mu=0, sigma=0.05)
#M += circle(shape, 8) # 40*circle(shape, 10)  # You can create a mask with circle, ellipse or gaussian fcts from utils
# M = ellipse(shape,30,80,45) + M
# M = np.mean(open_fits("../DISK/PDS201/"+"/psf"), axis=0)
# If you have wish to define the initialisation yourself
L0 = open_fits(datadir+"/L_rexpaco")+open_fits(datadir+"/L2")
X0 = open_fits(datadir+"/X_rexpaco")+open_fits(datadir+"/X2")
estimator.set_init(X0=X0/2, L0=L0/2)
estimator.configR2(Msk=M, mode="mask", penaliz="Both", invert=True)
estimator.configR1(mode="smooth", p_L=3, p_X=2)
estimator.configR3(mode="l1", p_L=0, p_X=0.1)
#  N.B : or you can juste trust the default parameters and don't call thoses methodes

# %% -------------------------------------

# In case you didn't provid a savedir before and you don't want it to be print at workingdir
# estimator.set_savedir(datadir+param['suffix'])

# Estimation of stellar halo
# estimator.estimate_halos(full_output=True, save=True)

if True :
    if param['estimI'] :  L_est, X_est, flux = estimator.estimate(**param, save=datadir, gif=False, verbose=True)
    else : L_est, X_est = estimator.estimate(**param, save=datadir, gif=False, verbose=True)
    # N.B : You can stop estimation with esc/ctrl+C. It will terminate properly and results will be return & save

    # Complete results are stored in the estimator ...
    res = estimator.res
    
    # You can access easily to what you need with get methods  :
    L0, X0 = estimator.get_initialisation(save="./L0x0/")  # initialization

    estimator.get_residual(way="direct", save=True)  # Residual cube
    estimator.get_reconstruction(way="direct", save=True)  # Reconstruction
    #estimator.get_radial_prof(show=True, save=True)
    #estimator.get_evo_convergence(show=False, save=True)
    #estimator.get_flux(show=False, save=True)
    #estimator.get_rot_weight(show=False, save=True)
    # estimator.mustard_results()

