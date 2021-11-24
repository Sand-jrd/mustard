#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Tests of neo-mayo

@author: sand-jrd
"""

from neo_mayo import mayo_estimator
import numpy as np
import torch

# %% -------------------------------------
# Test mayo estimator initialisation

# Choose where to get datas
datadir = "./example-data"

Test_greed = False
Test_model = False
Test_mayo  = True

regul_weight = 0
delta = 1e4
maxiter = 7

# %% -------------------------------------

# init the estimator and set variable
estimator = mayo_estimator(datadir, delta=delta, rot="fft", loss="mse")
shape = estimator.shape
model = estimator.model
angles, psf, science_data = estimator.get_science_data()

# %% Test Greed

L_ini, X_ini = estimator.initialisation(from_dir=datadir + "/L0X0")

if Test_greed:
    L_ini, X_ini = estimator.initialisation(save=datadir + "/L0X0", max_comp=4, nb_iter=10)

    # ___________________
    # Show results

    import matplotlib.pyplot as plt

    plt.figure("Greed Results")
    args = {"cmap": "magma", "vmax": np.percentile(L_ini, 98), "vmin": np.percentile(L_ini, 0)}

    plt.subplot(1, 2, 1), plt.imshow(L_ini, **args), plt.title("L estimation from pcait init")
    plt.subplot(1, 2, 2), plt.imshow(X_ini, **args), plt.title("X estimation from pcait init")


# %% -------------------------------------
# Test Forward model

def test_model(L_ini, X_ini):
    L_ini_tensor = torch.unsqueeze(torch.from_numpy(L_ini), 0)
    X_ini_tensor = torch.unsqueeze(torch.from_numpy(X_ini), 0)

    Y_tensor = model.forward_ADI(L_ini_tensor, X_ini_tensor)
    Y = Y_tensor.detach().numpy()[:, 0, :, :]
    return Y, L_ini_tensor, X_ini_tensor


if Test_model:

    Y, _, _ = test_model(L_ini, X_ini)

    # ___________________
    # Show results

    import matplotlib.pyplot as plt

    plt.figure("Forward model")
    args = {"cmap": "magma", "vmax": np.percentile(science_data, 98), "vmin": np.percentile(science_data, 0)}
    frame_ID = 1

    win_siz = 4
    frame_pas = model.nb_frame // (win_siz // 2 * (win_siz - 1))
    for ii in range(win_siz // 2 * (win_siz - 1)):
        plt.subplot(win_siz, win_siz, 2 * ii + 1), plt.imshow(Y[ii * frame_pas], **args), plt.title(
            "Y from forward model, img°" + str(ii))
        plt.subplot(win_siz, win_siz, 2 * ii + 2), plt.imshow(science_data[ii * frame_pas], **args), plt.title(
            "Real Y from science data, img°" + str(ii))

    plt.subplot(win_siz, 2, 2 * win_siz - 1), plt.imshow(L_ini, **args), plt.title("Given L")
    plt.subplot(win_siz, 2, 2 * win_siz), plt.imshow(X_ini, **args), plt.title("Given X")


# %% -------------------------------------
# Test Estimations

from vip_hci.var import frame_filter_lowpass
from vip_hci.preproc import frame_rotate


if Test_mayo:

    L_est, X_est = estimator.estimate(R_weights=regul_weight,
                                      maxiter=maxiter,
                                      save=False, gif=False, verbose=True)
    
    # Complete results are stored in the estimator
    res = estimator.res 

    # ___________________
    # Show results

    import matplotlib.pyplot as plt

    # ___________________
    plt.figure("Test mayo : ")
    args = {"cmap": "magma", "vmax": np.percentile(L_ini, 98)}

    plt.subplot(2, 3, 1), plt.imshow(L_est, **args), plt.title("L estimation")
    plt.subplot(2, 3, 4), plt.imshow(X_est, **args), plt.title("X estimation")
    plt.subplot(2, 3, 2), plt.imshow(L_ini, **args), plt.title("L estimation from pcait init")
    plt.subplot(2, 3, 5), plt.imshow(X_ini, **args), plt.title("X estimation from pcait init")
    plt.subplot(2, 3, 3), plt.imshow(abs(L_ini - L_est), **args), plt.title("Diff L")
    plt.subplot(2, 3, 6), plt.imshow(abs(X_ini - X_est), **args), plt.title("Diff X")

    # ___________________
    plt.figure("Loss evolutions : ")
    ex_frame = 30
    args = {"cmap": "magma", "vmax": np.percentile(science_data[ex_frame], 98), "vmin": np.percentile(science_data[ex_frame], 0)}

    Y_ini, L_t_ini, X_t_ini = test_model(L_ini, X_ini)
    Y_est, L_t_est, X_t_est = test_model(L_est, X_est)

    plt.subplot(2, 3, 1), plt.imshow(Y_ini[ex_frame], **args), plt.title("Y from ini, frame " + str(ex_frame))
    plt.subplot(2, 3, 2), plt.imshow(Y_est[ex_frame], **args), plt.title("Y from estimation, frame " + str(ex_frame))
    plt.subplot(2, 3, 3), plt.imshow(science_data[ex_frame], **args), plt.title("Y from science data, frame " + str(ex_frame))
    plt.subplot(2, 2, 3), plt.imshow(abs(science_data[ex_frame] - Y_ini[ex_frame]), **args), plt.title("Diff between Y_ini/Y_science; loos ={:.2e}".format(res['loss_evo'][0]))
    plt.subplot(2, 2, 4), plt.imshow(abs(science_data[ex_frame] - Y_est[ex_frame]), **args), plt.title("Diff between Y_est/Y_science; loss ={:.2e}".format(res['loss_evo'][-1]))

    # ___________________
    plt.figure("Test truc")
    plt.subplot(1, 3, 1), plt.imshow(science_data[0], **args), plt.title("Data ")
    plt.subplot(1, 3, 2), plt.imshow(L_est, **args), plt.title("Estimated L ")
    plt.subplot(1, 3, 3), plt.imshow(science_data[0]-L_est, **args), plt.title("Data - estimated L ")

    # ___________________
    plt.figure("compare")
    plt.subplot(1, 3, 1), plt.imshow(science_data[35]-L_est, **args), plt.title("Data - estimated L ")
    plt.subplot(1, 3, 2), plt.imshow(X_est, **args), plt.title("Estimated X ")
    plt.subplot(1, 3, 3), plt.imshow(X_ini, **args), plt.title("X with iterative PCA ")

    # ___________________
    plt.figure("Remove noise")
    args = {"cmap": "magma", "vmax": np.percentile(science_data[35]-L_est, 99), "vmin": np.percentile(X_est, 0)}

    derot_Xs_est = np.zeros(science_data.shape)
    for frame in range(model.nb_frame):
        derot_Xs_est[frame] = frame_rotate(science_data[frame]-L_est, -angles[frame])

    plt.subplot(2, 2, 1), plt.imshow(X_est, **args), plt.title("Estimated X ")
    plt.subplot(2, 2, 2), plt.imshow(science_data[0]-L_est, **args), plt.title("Data - L ")
    plt.subplot(2, 2, 3), plt.imshow(frame_filter_lowpass(science_data[0]-L_est, kernel_sz=3), **args), plt.title("Filtered (Data - L)")
    plt.subplot(2, 2, 4), plt.imshow(np.mean(derot_Xs_est, axis=0), **args), plt.title("Mean (Data - L)  ")