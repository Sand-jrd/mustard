#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:27:42 2021

______________________________

| algortihms used in neo-mayo |
______________________________

@author: sand-jrd
"""

from vip_hci.pca import pca_fullfr
from vip_hci.preproc import frame_rotate
from vip_hci.itpca import pca_it

import numpy as np


# For verbose


# %% Iterative PCA aka GREED

def init_estimate(cube, angle_list, mode=None, ncomp_step=1, n_it=10, **kwarg):
    """ Iterative PCA as describe in Mayo """

    L_k = np.zeros(cube.shape)
    X_k = np.zeros(cube.shape)
    nb_frame = cube.shape[0]

    if mode == "sand":

        for iter_k in range(n_it):
            for nb_comp in range(ncomp_step):

                res = pca_fullfr.pca(L_k, angle_list, ncomp=nb_comp, verbose=False)

                # Since res is derotated we have to rerotate it ...
                for frame_id in range(nb_frame):
                    frame_id_rot = frame_rotate(res, angle_list[frame_id])
                    L_k[frame_id] = cube[frame_id] - frame_id_rot.clip(min=0)
                    X_k[frame_id] = frame_id_rot.clip(min=0)

    if mode == "pca":

        res = pca_fullfr.pca(L_k, angle_list, ncomp=ncomp_step, verbose=False)

        for frame_id in range(nb_frame):
            frame_id_rot = frame_rotate(res, angle_list[frame_id])
            L_k[frame_id] = cube[frame_id] - (frame_id_rot.clip(min=0))

    else:

        res = pca_it(cube, angle_list,
                     mode=mode,
                     ncomp_step=ncomp_step,
                     n_it=n_it,
                     verbose=False, **kwarg)

        for frame_id in range(nb_frame):
            frame_id_rot = frame_rotate(res, angle_list[frame_id])
            L_k[frame_id] = cube[frame_id] - (frame_id_rot.clip(min=0))

    return np.median(L_k, axis=0).clip(min=0), res.clip(min=0)


# %% Minimizer with toch

import torch
import torch.optim as optim
from neo_mayo.model import regul_L_torch, regul_X
from neo_mayo.utils import var_inmatrix, var_inline, print_iter


def torch_minimiz(fun, x0, args, nb_iter=10, gif=True, **kwarg):
    if "options" in kwarg.keys():
        options = kwarg["options"]
        if "maxiter" in options.keys(): nb_iter = options["maxiter"]

    model = args[0]
    constantes = args[1]

    R_weight = constantes["hyper_p"]

    model.adapt_torch(delta=constantes["delta"])

    science_data = torch.unsqueeze(torch.from_numpy(constantes["science_data"]), 1).float()
    science_data.requires_grad = False

    constantes["L_proj"] = torch.unsqueeze(torch.from_numpy(constantes["L_proj"]), 0).float()
    constantes["L_proj"].requires_grad = False

    L0, X0 = var_inmatrix(x0, model.frame_shape[0])

    L0 = model.mask * torch.unsqueeze(torch.from_numpy(L0), 0).float()
    L0.requires_grad = True

    X0 = model.mask * torch.unsqueeze(torch.from_numpy(X0), 0).float()
    X0.requires_grad = True

    optimizer = optim.LBFGS([L0, X0])

    Y = model.forward_torch_ADI(L0, X0)
    loss = model.hub_loss(Y, science_data)

    def closure():
        global loss
        optimizer.zero_grad()
        Y = model.forward_torch_ADI(L0, X0)
        R = R_weight * regul_X(X0)
        loss = model.hub_loss(Y, science_data) + R
        loss.backward()
        print("Iteration n°" + str(ii) + " {:.2e}".format(model.hub_loss(Y, science_data) + R))
        return loss

    for ii in range(nb_iter):
        if gif: print_iter(L0, X0, ii, loss)
        optimizer.step(closure)

    # Back to numpy array
    L0 = L0.detach().numpy()[0, :, :]
    X0 = X0.detach().numpy()[0, :, :]

    res = dict()
    res["state"] = optimizer.state
    res['x'] = var_inline(abs(L0), abs(X0))

    return res


# %% Other ALGO


def compute_L_proj(L_est):
    """ Compute a L projector """

    U, S, V = np.linalg.svd(L_est)
    proj = U @ np.transpose(U)

    return proj
