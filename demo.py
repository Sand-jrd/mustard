#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Usage example of neo-mayo

@author: sand-jrd
"""

from neo_mayo import mayo_estimator
import numpy as np
import torch
from neo_mayo.utils import unpack_science_datadir

# %% Test mayo estimator initialisation

# Choose where to get datas
datadir = "./example-data"

# init the estimator
estimator = mayo_estimator(datadir)
angles, psf, science_data = estimator.get_science_data()

Test_greed = False
Test_model = False
Test_proj  = False
Test_mayo  = True

minimizer    = "torch"     #  Possible minimizer : {"torch", "SLSQP", "L-BFGS-B", "minimize_parallel"}
regul_weight = 1
delta        = 1e7
iter         = 5

# %% Test Greed

L_ini, X_ini = estimator.initalisation(from_dir=datadir+"/L0X0")

if Test_greed :

    L_ini, X_ini = estimator.initalisation(save=datadir + "/L0X0",max_comp=4,nb_iter=10)

    # ___________________
    # Show results

    import matplotlib.pyplot as plt

    plt.figure("Greed Results")
    args = {"cmap" : "magma", "vmax" : np.percentile(L_ini,98),"vmin" : np.percentile(L_ini,0)}

    plt.subplot(1,2,1),plt.imshow(L_ini, **args),plt.title("L estimation from pcait init")
    plt.subplot(1,2,2),plt.imshow(X_ini, **args),plt.title("X estimation from pcait init")


# %% Test Forward model

if Test_model :

    shape = estimator.shape
    model = estimator.model

    if minimizer=="torch" :
        estimator.model.adapt_torch()
        L_ini_tensor = torch.unsqueeze(torch.from_numpy(L_ini), 0)
        X_ini_tensor = torch.unsqueeze(torch.from_numpy(X_ini), 0)

        Y_tensor = model.forward_torch_ADI(L_ini_tensor,X_ini_tensor)
        Y = Y_tensor.detach().numpy()[:,0,:,:]

    else :
        Y = model.forward_ADI(L_ini,X_ini)

    # ___________________
    # Show results

    import matplotlib.pyplot as plt

    plt.figure("Forward model")
    args = {"cmap" : "magma", "vmax" : np.percentile(science_data,98), "vmin" : np.percentile(science_data,0)}
    frame_ID = 1

    win_siz   = 4
    frame_pas = model.nb_frame // (win_siz//2*(win_siz-1))
    for ii in range(win_siz//2*(win_siz-1)) :
        plt.subplot(win_siz,win_siz,2*ii+1),plt.imshow(Y[ii*frame_pas], **args),plt.title("Y from forward model, img°"+str(ii))
        plt.subplot(win_siz,win_siz,2*ii+2),plt.imshow(science_data[ii*frame_pas], **args),plt.title("Real Y from science data, img°"+str(ii))

    plt.subplot(win_siz,2,2*win_siz-1),plt.imshow(L_ini, **args),plt.title("Given L")
    plt.subplot(win_siz,2,2*win_siz),plt.imshow(X_ini, **args),plt.title("Given X")

# %% Test compute L_projections
from neo_mayo.algo import compute_L_proj

if Test_proj :
    Proj = compute_L_proj(L_ini)
    print("Loss projection : {:.2e}".format( np.sum((L_ini - (Proj @ L_ini))**2) ) )


# %% Test Estimations

if Test_mayo :

    L_est, X_est = estimator.estimate(delta=delta,
                                      hyper_p=regul_weight,
                                      minimizer=minimizer,
                                      options={"maxiter":iter,"iprint":1})

    # ___________________
    # Show results

    import matplotlib.pyplot as plt

    plt.figure("Test mayo : " + minimizer)
    args = {"cmap" : "magma", "vmax" : np.percentile(L_est,98)}

    for frame_ID in range(2) :
        plt.subplot(2, 3, 1),plt.imshow(L_est, **args),plt.title("L estimation")
        plt.subplot(2, 3, 4),plt.imshow(X_est, **args),plt.title("X estimation")
        plt.subplot(2, 3, 2), plt.imshow(L_ini, **args), plt.title("L estimation from pcait init")
        plt.subplot(2, 3, 5), plt.imshow(X_ini, **args), plt.title("X estimation from pcait init")
        plt.subplot(2, 3, 3), plt.imshow(abs(L_ini - L_est), **args), plt.title("Diff L")
        plt.subplot(2, 3, 6), plt.imshow(abs(X_ini - X_est), **args), plt.title("Diff X")