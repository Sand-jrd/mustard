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
Test_model = True
Test_proj = False
Test_mayo = False

minimizer = "torch"  # Possible minimizer : {"torch", "SLSQP", "L-BFGS-B", "minimize_parallel"}
regul_weight = 0
delta = 1e4
estim_iter = 7

# %% -------------------------------------

# init the estimator and set variable
estimator = mayo_estimator(datadir)
shape = estimator.shape
model = estimator.model
angles, psf, science_data = estimator.get_science_data()

# %% Test Greed

L_ini, X_ini = estimator.initalisation(from_dir=datadir + "/L0X0")

if Test_greed:
    L_ini, X_ini = estimator.initalisation(save=datadir + "/L0X0", max_comp=4, nb_iter=10)

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
    if minimizer == "torch":
        estimator.model.adapt_torch()
        L_ini_tensor = torch.unsqueeze(torch.from_numpy(L_ini), 0)
        X_ini_tensor = torch.unsqueeze(torch.from_numpy(X_ini), 0)

        Y_tensor = model.forward_torch_ADI(L_ini_tensor, X_ini_tensor)
        Y = Y_tensor.detach().numpy()[:, 0, :, :]
        return Y, L_ini_tensor, X_ini_tensor

    else:
        return model.forward_ADI(L_ini, X_ini)


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
# Test compute L_projections
from neo_mayo.algo import compute_L_proj

if Test_proj:
    Proj = compute_L_proj(L_ini)
    print("Loss projection : {:.2e}".format(np.sum((L_ini - (Proj @ L_ini)) ** 2)))


# %% -------------------------------------
# Test Estimations

from neo_mayo.model import adi_model_loss, adi_model_loss_torch, regul_X
from neo_mayo.utils import sobel_tensor_conv
from vip_hci.var import frame_filter_lowpass
from vip_hci.preproc import frame_rotate


if Test_mayo:

    L_est, X_est = estimator.estimate(delta=delta,
                                      hyper_p=regul_weight,
                                      minimizer=minimizer,
                                      options={"maxiter": estim_iter, "iprint": 1, "disp" : 1})

    # ___________________
    # Show results

    import matplotlib.pyplot as plt

    plt.figure("Test mayo : " + minimizer)
    args = {"cmap": "magma", "vmax": np.percentile(L_ini, 98)}

    plt.subplot(2, 3, 1), plt.imshow(L_est, **args), plt.title("L estimation")
    plt.subplot(2, 3, 4), plt.imshow(X_est, **args), plt.title("X estimation")
    plt.subplot(2, 3, 2), plt.imshow(L_ini, **args), plt.title("L estimation from pcait init")
    plt.subplot(2, 3, 5), plt.imshow(X_ini, **args), plt.title("X estimation from pcait init")
    plt.subplot(2, 3, 3), plt.imshow(abs(L_ini - L_est), **args), plt.title("Diff L")
    plt.subplot(2, 3, 6), plt.imshow(abs(X_ini - X_est), **args), plt.title("Diff X")

    plt.figure("Loss evolutions : " + minimizer)
    ex_frame = 30
    args = {"cmap": "magma", "vmax": np.percentile(science_data[ex_frame], 98), "vmin": np.percentile(science_data[ex_frame], 0)}

    if minimizer == "torch":
        Y_ini, L_t_ini, X_t_ini = test_model(L_ini, X_ini)
        Y_est, L_t_est, X_t_est = test_model(L_est, X_est)
        Xregul_ini = regul_X(X_t_ini)
        Xregul_est = regul_X(X_t_est)
        loss_ini = adi_model_loss_torch(model, L_t_ini, X_t_ini, estimator.constantes)
        loss_est = adi_model_loss_torch(model, L_t_est, X_t_est, estimator.constantes)
    else:
        Y_ini = test_model(L_ini, X_ini)
        Y_est = test_model(L_est, X_est)
        loss_ini = adi_model_loss(model, L_ini, X_ini, estimator.constantes)
        loss_est = adi_model_loss(model, L_est, X_est, estimator.constantes)

    plt.subplot(2, 3, 1), plt.imshow(Y_ini[ex_frame], **args), plt.title("Y from ini, frame " + str(ex_frame))
    plt.subplot(2, 3, 2), plt.imshow(Y_est[ex_frame], **args), plt.title("Y from estimation, frame " + str(ex_frame))
    plt.subplot(2, 3, 3), plt.imshow(science_data[ex_frame], **args), plt.title("Y from science data, frame " + str(ex_frame))
    plt.subplot(2, 2, 3), plt.imshow(abs(science_data[ex_frame] - Y_ini[ex_frame]), **args), plt.title("Diff between Y_ini/Y_science; loos ={:.2e}".format(loss_ini))
    plt.subplot(2, 2, 4), plt.imshow(abs(science_data[ex_frame] - Y_est[ex_frame]), **args), plt.title("Diff between Y_est/Y_science; loss ={:.2e}".format(loss_est))

    plt.figure("Test regul")
    plt.subplot(2, 1, 1), plt.imshow(sobel_tensor_conv(X_t_ini).detach().numpy()[0,0,:,:].clip(min=0), **args), plt.title("Regul at ini = {:.2e}".format(Xregul_ini))
    plt.subplot(2, 1, 2), plt.imshow(sobel_tensor_conv(X_t_est).detach().numpy()[0,0,:,:].clip(min=0), **args), plt.title("Regul at est = {:.2e}".format(Xregul_est))

    plt.figure("Test truc")
    plt.subplot(1, 3, 1), plt.imshow(science_data[0], **args), plt.title("Data ")
    plt.subplot(1, 3, 2), plt.imshow(L_est, **args), plt.title("Estimated L ")
    plt.subplot(1, 3, 3), plt.imshow(science_data[0]-L_est, **args), plt.title("Data - estimated L ")

    plt.figure("compare")
    plt.subplot(1, 3, 1), plt.imshow(science_data[35]-L_est, **args), plt.title("Data - estimated L ")
    plt.subplot(1, 3, 2), plt.imshow(X_est, **args), plt.title("Estimated X ")
    plt.subplot(1, 3, 3), plt.imshow(X_ini, **args), plt.title("X with iterative PCA ")


derot_Xs_est = np.zeros(science_data.shape)
for frame in range(model.nb_frame):
    derot_Xs_est[frame] = frame_rotate(science_data[frame]-L_est,-angles[frame])

    args = {"cmap": "magma", "vmax": np.percentile(science_data[35]-L_est, 99), "vmin": np.percentile(X_est, 0)}

    plt.figure("Remove noise")
    plt.subplot(2, 2, 1), plt.imshow(X_est, **args), plt.title("Estimated X ")
    plt.subplot(2, 2, 2), plt.imshow(science_data[0]-L_est, **args), plt.title("Data - L ")
    plt.subplot(2, 2, 3), plt.imshow(frame_filter_lowpass(science_data[0]-L_est,kernel_sz=3), **args), plt.title("Filterd (Data - L)")
    plt.subplot(2, 2, 4), plt.imshow(np.mean(derot_Xs_est,axis=0), **args), plt.title("Mean (Data - L)  ")


