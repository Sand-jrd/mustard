#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:37:16 2021

______________________________

| Utils Function for neo_mayo  |
______________________________


@author: sand-jrd
"""
# Misc
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

# File management
from vip_hci.fits import open_fits
import json, glob, os


# %% Create patterns

def circle(shape, r, offset=0.5):
    """ Create circle of 1 in a 2D matrix of zeros"
       
       Parameters
       ----------

       shape : tuple
           shape x,y of the matrix
       
       r : float
           radius of the circle
       offset : (optional) float
           offset from the center
       
       Returns
       -------
       M : ndarray
           Zeros matrix with a circle filled with ones
       
    """
    M = np.zeros(shape)
    w, l = shape
    for x in range(0, w):
        for y in range(0, l):
            if pow(x - (w / 2) + offset, 2) + pow(y - (l / 2) + offset, 2) < pow(r, 2):
                M[x, y] = 1
    return M


# %% Manage files

def unpack_science_datadir(datadir):
    #  Import data
    json_file = glob.glob(datadir + "/*.json")

    if len(json_file) == 0:
        raise AssertionError("Json file not found in in data folder : " + str(datadir))
    elif len(json_file) > 1:
        raise AssertionError("More than two json file found in data folder : " + str(datadir))

    with open(json_file[0], 'r') as read_data_info:
        data_info = json.load(read_data_info)

        # Checks if all required keys are here
    required_keys = ("cube", "angles", "psf")
    if not all([key in data_info.keys() for key in required_keys]):
        raise AssertionError("Data json info does not contained required keys")

    # Open fits
    angles = open_fits(datadir + "/" + data_info["angles"], verbose=False)
    psf = open_fits(datadir + "/" + data_info["psf"], verbose=False)
    if len(psf.shape) == 3: psf = psf[data_info["which_psf"]]

    science_data = open_fits(datadir + "/" + data_info["cube"], verbose=False)

    return angles, psf, science_data


def print_iter(L: torch.Tensor, x: torch.Tensor, bfgs_iter, loss):
    L_np = L.detach().numpy()[0, :, :]
    X_np = x.detach().numpy()[0, :, :]

    plt.ioff()
    plt.figure("Iterations plot temporary")

    plt.suptitle("Iteration n°" + str(bfgs_iter) + "\nLoss  = {:.6e}".format(loss))
    args = {"cmap": "magma", "vmax": np.percentile(L_np, 98), "vmin": np.percentile(L_np, 0)}

    plt.subplot(1, 2, 1), plt.imshow(L_np, **args), plt.title("L estimation from pcait init")
    plt.subplot(1, 2, 2), plt.imshow(X_np, **args), plt.title("X estimation from pcait init")

    plt.savefig("./iter/" + str(bfgs_iter) + ".png", pad_inches=0.5)
    plt.clf()


def iter_to_gif(save_gif='.', name="sim"):
    images = []
    plt.ion()

    if not os.path.isdir(save_gif): os.makedirs(save_gif)

    for file in sorted(glob.glob("./iter/*.png"), key=os.path.getmtime):
        images.append(Image.open(file))
    images[0].save(fp=save_gif + str(name) + ".gif", format='GIF', append_images=images, save_all=True, duration=500,
                   loop=0)

    ii = 0
    for file in sorted(glob.glob("./iter/*.png")):
        images[ii].close()
        try:
            os.remove(file)
        except PermissionError:
            print(
                "Can't delete saves for gif. A process might not have closed properly. "
                "\nIgnore deletion \nI highly suggest you to delete it.")
        ii += 1
