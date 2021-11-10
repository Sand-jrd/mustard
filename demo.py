#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Usage axemple of neo-mayo

@author: sand-jrd
"""

from neo_mayo import mayo_estimator
import numpy as np

# %% Test mayo estimator initialisation

# Choose where to get datas
datadir = "./example-data"

# init the estimator
estimator = mayo_estimator(datadir)

Test_greed = False
Test_model = False
Test_proj  = False
Test_mayo  = True

# %% Test Greed

if Test_greed :

    from neo_mayo.algo import Greed
    from neo_mayo.utils import unpack_science_datadir
    
    angles,psf,science_data = unpack_science_datadir(datadir)
    L_est,X_est = Greed(science_data, angles,max_comp=2,nb_iter=10)
    
    # ___________________
    # Show results
    
    import matplotlib.pyplot as plt
    
    plt.figure("Greed Results")
    for frame_ID in range(2) :
        args = {"cmap" : "magma", "vmax" : np.percentile(L_est,98)}
        plt.subplot(2,2,2*frame_ID+1),plt.imshow(L_est[frame_ID], **args),plt.title("L estimation; frame n°"+str(frame_ID))
        plt.subplot(2,2,2*frame_ID+2),plt.imshow(X_est[frame_ID], **args),plt.title("L estimation; frame n°"+str(frame_ID))


# %% Test Forward model

if Test_model :
    
    shape = estimator.shape
    model = estimator.model
    
    Y = model.forward_ADI(L_est,X_est)
    
    # ___________________
    # Show results
    
    import matplotlib.pyplot as plt
    
    plt.figure("Forward model")
    args = {"cmap" : "magma", "vmax" : np.percentile(Y,98)}

    plt.subplot(2,3,1),plt.imshow(Y[frame_ID], **args),plt.title("Y from forward model")
    plt.subplot(2,3,2),plt.imshow(science_data[frame_ID], **args),plt.title("Real Y from science data")
    plt.subplot(2,3,3),plt.imshow(science_data[frame_ID]-Y[frame_ID], **args),plt.title("Difference")


    plt.subplot(2,2,3),plt.imshow(L_est[frame_ID], **args),plt.title("Given L")
    plt.subplot(2,2,4),plt.imshow(X_est[frame_ID], **args),plt.title("Given X")

# %% Test compute L_projections
from algo import compute_L_proj

if Test_proj :
    Proj = compute_L_proj(L_est)
    print("Loss projection : {:.2e}".format( np.sum((L_est - (Proj @ L_est))**2) ) )


# %% Test Estimations

if Test_mayo : 
   
    L_est, X_est = estimator.estimate(hyper_p=1,options={"maxiter":250})
    
    # ___________________
    # Show results
    
    import matplotlib.pyplot as plt
    
    plt.figure("Test mayo")
    args = {"cmap" : "magma", "vmax" : np.percentile(L_est,98)}

    for frame_ID in range(2) :
        plt.subplot(2,2,2*frame_ID+1),plt.imshow(L_est[frame_ID], **args),plt.title("L estimation; frame n°"+str(frame_ID))
        plt.subplot(2,2,2*frame_ID+2),plt.imshow(X_est[frame_ID], **args),plt.title("X estimation; frame n°"+str(frame_ID))
