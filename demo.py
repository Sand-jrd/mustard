#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Tests of neo-mayo

@author: sand-jrd
"""
import matplotlib.pyplot as plt
from neo_mayo import mayo_estimator
from neo_mayo.utils import ellipse,circle
from vip_hci.fits import open_fits
from vip_hci.var import frame_filter_lowpass
from vip_hci.preproc import frame_rotate
import numpy as np
import glob
import torch

# %% -------------------------------------
# Test mayo estimator initialisation

# Choose where to get datas

datadir = "./example-data/"
#datadir = "../PDS70-neomayo/095.C-0298A/H2/"

Test_ini    = False
Test_model  = False
Test_regul  = False
Test_mayo   = False
i_have_time = False # Extra outputs
show_mask   = True

param = {'w_r'   : 0.2,
        'w_r2'   : 0.2,
        'kactiv' : None,
        'kdactiv': None,
        'estimI' : True,
        'maxiter': 30}

# init the estimator and set variable
estimator = mayo_estimator(datadir, rot="fft", loss="mse", regul="smooth", Gframes= list(range(1,19)))
shape = estimator.shape
model = estimator.model
angles, science_data = estimator.get_science_data()
# %% -------------------------------------

# init R2 regularization (optional)
M =  open_fits("/Users/sand-jrd/Desktop/DPI/denoise/1100.C-0481T_1.fits").clip(0)
mid = M.shape[0]//2; size = model.frame_shape[0]//2;
# M = M[mid-size:mid+size,mid-size:mid+size]                            # if frame too big
M0 = np.zeros(model.frame_shape); M0[size-mid:size+mid,size-mid:size+mid]=M;M=M0   # if frame too small
M = np.max(science_data[0]) * M/np.max(M)
# M = ellipse(model.frame_shape,60, 40, 5) \
   # - circle(model.frame_shape,10)
R2_param = { 'M'    : M,
           'mode'   : "dist",
           'penaliz': "X",
           'invert' : True }

estimator.configR2(**R2_param)

if show_mask :
    exFrame = frame_rotate(science_data[0], -angles[0])
    exFrame = open_fits(datadir + "/X_est.fits")
    args = {"cmap": "magma", "vmax": np.percentile(science_data[0], 98)}


    plt.figure("the mask")
    plt.subplot(131),plt.imshow(M,**args),plt.title("Mask")
    plt.subplot(132),plt.imshow(exFrame,**args),plt.title("X")
    if R2_param['mode'] == "mask": plt.subplot(133),plt.imshow(np.abs(M*exFrame),**args),plt.title("Mask * X")
    if R2_param['mode'] == "dist": plt.subplot(133),plt.imshow(np.abs(M-exFrame),**args),plt.title("Mask - X")





# %% -------------------------------------

# %% Test Greed

if Test_ini:
    L_ini, X_ini = estimator.initialisation(save=datadir + "/L0X0", mode="pca", max_comp=1, nb_iter=10)

    # ___________________
    # Show results

    plt.figure("Greed Results")
    args = {"cmap": "magma", "vmax": np.percentile(L_ini, 98), "vmin": np.percentile(L_ini, 0)}

    plt.subplot(1, 2, 1), plt.imshow(L_ini, **args), plt.title("L estimation from pcait init")
    plt.subplot(1, 2, 2), plt.imshow(X_ini, **args), plt.title("X estimation from pcait init")

else : L_ini, X_ini = estimator.initialisation(from_dir=datadir + "/L0X0")

# %% -------------------------------------
# Test Forward model

def test_model(L_ini, X_ini, flux=torch.ones(model.nb_frame-1)):
    L_ini_tensor = torch.unsqueeze(torch.from_numpy(L_ini), 0)
    X_ini_tensor = torch.unsqueeze(torch.from_numpy(X_ini), 0)

    Y_tensor = model.forward_ADI(L_ini_tensor, X_ini_tensor, flux)
    Y = Y_tensor.detach().numpy()[:, 0, :, :]
    return Y, L_ini_tensor, X_ini_tensor


if Test_model:

    Y, _, _ = test_model(L_ini, X_ini)

    # ___________________
    # Show results

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


if Test_regul :

    from neo_mayo.algo import sobel_tensor_conv

    X_ini_tensor = torch.unsqueeze(torch.from_numpy(X_ini), 0).double()
    X_tensor = sobel_tensor_conv(X_ini_tensor)
    X_r = X_tensor.detach().numpy()[0, 0, :, :]

    # ___________________
    # Show results

    plt.figure("Regul")
    args = {"cmap": "magma", "vmax": np.percentile(X_r, 100), "vmin": np.percentile(X_r, 0)}

    plt.imshow(X_r, **args), plt.title("Loss per pixels; square sum = {:.2e}".format(torch.sum(X_tensor**2)))


if Test_mayo:

    # L_est, X_est = estimator.estimate(**param,save=False, gif=True, verbose=True)
    L_est, X_est, flux = estimator.estimate(**param, save=datadir, gif=True, verbose=True)
    
    # Complete results are stored in the estimator
    res = estimator.res 

    # ___________________
    # Show results

    plt.figure("Loss evolutions : ")
    ex_frame = 5
    args = {"cmap": "magma", "vmax": np.percentile(science_data[ex_frame], 98), "vmin": np.percentile(science_data[ex_frame], 0)}

    Y_ini, L_t_ini, X_t_ini = test_model(L_ini, X_ini, flux)
    Y_est, L_t_est, X_t_est = test_model(L_est, X_est, flux)

    plt.subplot(2, 3, 1), plt.imshow(Y_ini[ex_frame], **args), plt.title("Y from ini, frame " + str(ex_frame))
    plt.subplot(2, 3, 2), plt.imshow(Y_est[ex_frame], **args), plt.title("Y from estimation, frame " + str(ex_frame))
    plt.subplot(2, 3, 3), plt.imshow(science_data[ex_frame], **args), plt.title("Y from science data, frame " + str(ex_frame))
    plt.subplot(2, 2, 3), plt.imshow(abs(science_data[ex_frame] - Y_ini[ex_frame]), **args), plt.title("Diff between Y_ini/Y_science; loos ={:.2e}".format(res['loss_evo'][0]))
    plt.subplot(2, 2, 4), plt.imshow(abs(science_data[ex_frame] - Y_est[ex_frame]), **args), plt.title("Diff between Y_est/Y_science; loss ={:.2e}".format(res['loss_evo'][-1]))
    plt.savefig(datadir + ".png", pad_inches=0.5)

    # ___________________
    plt.figure("compare")
    plt.subplot(3, 2, 1), plt.imshow(X_est, **args), plt.title("Estimated X ")
    plt.subplot(3, 2, 2), plt.imshow(X_ini, **args), plt.title("X with iterative PCA ")
    plt.subplot(3, 2, 3), plt.imshow(L_est, **args), plt.title("Estimated L ")
    plt.subplot(3, 2, 4), plt.imshow(L_ini, **args), plt.title("L with iterative PCA ")
    plt.subplot(3, 1, 3), plt.imshow(science_data[ex_frame], **args), plt.title("Science data")
    plt.savefig(datadir + ".png", pad_inches=0.5)

if i_have_time :

    plt.figure("Test mayo : ")
    args = {"cmap": "magma", "vmax": np.percentile(L_ini, 98)}

    plt.subplot(2, 3, 1), plt.imshow(L_est, **args), plt.title("L estimation")
    plt.subplot(2, 3, 4), plt.imshow(X_est, **args), plt.title("X estimation")
    plt.subplot(2, 3, 2), plt.imshow(L_ini, **args), plt.title("L estimation from pcait init")
    plt.subplot(2, 3, 5), plt.imshow(X_ini, **args), plt.title("X estimation from pcait init")
    plt.subplot(2, 3, 3), plt.imshow(abs(L_ini - L_est), **args), plt.title("Diff L")
    plt.subplot(2, 3, 6), plt.imshow(abs(X_ini - X_est), **args), plt.title("Diff X")

    # ___________________
    plt.figure("Remove noise")
    args = {"cmap": "magma", "vmax": np.percentile(science_data, 98), "vmin": np.percentile(science_data, 0)}

    derot_Xs_est = np.zeros(science_data.shape)
    for frame in range(model.nb_frame):
        derot_Xs_est[frame] = frame_rotate(science_data[frame]-L_est, -angles[frame])

    plt.subplot(2, 2, 1), plt.imshow(abs(X_est), **args), plt.title("Estimated X ")
    plt.subplot(2, 2, 2), plt.imshow(science_data[0]-L_est, **args), plt.title("Data - L ")
    plt.subplot(2, 2, 3), plt.imshow(frame_filter_lowpass(science_data[0]-L_est, kernel_sz=3), **args), plt.title("Filtered (Data - L)")
    plt.subplot(2, 2, 4), plt.imshow(np.mean(derot_Xs_est, axis=0), **args), plt.title("Mean (Data - L)  ")