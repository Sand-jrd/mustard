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
from os import mkdir
from os.path import isdir
from datetime import datetime

BoxStyle = dict(boxstyle='square', pad=1.3, mutation_scale=0.01, facecolor='indigo', alpha=0.65)
FontStyle = dict(family= 'sans', color= 'beige', weight='normal', size= 12)
estimBox = dict(boxstyle='square', facecolor='beige', edgecolor='indigo', alpha=0.65)


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


def ellipse(shape,small_ax, big_ax, rotation, off_center=[0, 0]):
    """ Create ellipse of 1 in a 2D matrix of zeros"

       Parameters
       ----------
       shape : tuple
           shape x,y of the matrix

       small_ax : float
            radius of small ax of the ellipse

       big_ax : float
            radius of small ax of the big_ax

        rotation : float
            rotation of the ellipse

        off_center : list or tuple of 2 float
            shift x,y of the center of the ellipse

       Returns
       -------
       M : ndarray
           Zeros matrix with a circle filled with ones
        """

    mid =  np.array(shape) // 2 - off_center
    M = np.zeros(shape)
    w, l = shape
    rotation = np.deg2rad(rotation)

    def isInEllipse(x,y):
        term1 = (x - mid[0])*np.cos(rotation) +  (y - mid[1])*np.sin(rotation)
        term2 = (x - mid[0])*np.sin(rotation) - (y - mid[1])*np.cos(rotation)
        return (term1 / small_ax)**2 + (term2 / big_ax)**2  <= 1

    # Generate elips coordinate trace points
    for x in range(shape[0]):
        for y in range(shape[1]):
            if isInEllipse(x, y):
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
    required_keys = ("cube", "angles")
    if not all([key in data_info.keys() for key in required_keys]):
        raise AssertionError("Data json info does not contained required keys")

    # Open fits
    angles = open_fits(datadir + "/" + data_info["angles"], verbose=False)

    science_data = open_fits(datadir + "/" + data_info["cube"], verbose=False)

    return angles, science_data


def print_iter(L: torch.Tensor, x: torch.Tensor, bfgs_iter, loss, R1, R2, config, w_r, Ractiv, estimL, flux, datadir):
    L_np  = L.detach().numpy()[0, :, :]
    X_np  = x.detach().numpy()[0, :, :]
    fluxnp = flux.detach().numpy()

    plt.ioff()
    col = 3 if estimL else 2
    ratios = [5, 1, 2] if estimL else [3, 1]

    plt.subplots(col, 2, figsize=(16, 9), gridspec_kw={'height_ratios': ratios})

    plt.suptitle("Iteration n°" + str(bfgs_iter) + "\nLoss  = {:.6e}".format(loss))
    args = {"cmap": "magma", "vmax": np.percentile(L_np, 98), "vmin": np.percentile(L_np, 0)}

    if not bfgs_iter : title = "Initialisation"
    else : title = "Estimation of L and X"
    if estimL : title +=" and flux"
    title += " at iteration n°" + str(bfgs_iter)

    plt.suptitle(title)

    plt.subplot(col, 2, 1), plt.imshow(np.abs(L_np), **args), plt.title("L (starlight) ")
    plt.subplot(col, 2, 2), plt.imshow(np.abs(X_np), **args), plt.title("X (circonstellar light)")

    infos  = "\nMinimiz LBFGS with '"+str(config[1])+"' loss and '"+str(config[0])+"' regul" +\
             "\n w_r = {:.2f}".format(w_r)
    infos += ", R is activated" if Ractiv else ", R is deactivated"
    infos += "\n\nIteration n°" + str(bfgs_iter) + " - loss = {:.6e}".format(loss) +\
             "\n   R = {:.4e} ({:.0f}%)".format(R1, 100 * R1 / loss) + \
             "\n   R = {:.4e} ({:.0f}%)".format(R2, 100 * R2 / loss) + \
             "\n    J = {:.4e} ({:.0f}%) \n".format(loss, 100 * (loss-R1-R2) / loss )

    if estimL :

        plt.subplot2grid((col, 2), (1, 0), colspan=2)
        plt.bar(range(2, len(fluxnp)+2), fluxnp-1, color='beige', edgecolor="black"),plt.xticks(range(1,len(fluxnp)+2))
        plt.ylabel("I factor diff"), plt.xlabel("frameID"),plt.title("flux variations")

        LIinfo = "I min : " + str(np.min(fluxnp)) + "\nI max : " + str(np.max(fluxnp))
        plt.text(0, 0, LIinfo, bbox=estimBox, fontdict={'color':'indigo'})

    plt.subplot2grid((col,2), (2 if estimL else 1, 0), colspan=2)
    plt.axis('off'), plt.text(0.3, -0.2, infos, bbox=BoxStyle, fontdict=FontStyle)

    if not datadir : datadir = "."
    if not isdir(datadir + "/iter/"): mkdir(datadir + "/iter/")

    plt.savefig(datadir + "/iter/" + str(bfgs_iter) + ".png", pad_inches=0.5)
    plt.clf(), plt.close()


def iter_to_gif(save_gif, name="sim"):
    images = []
    plt.ion()

    date = datetime.now()
    if not save_gif: save_gif = "."

    double_init = True
    for file in sorted(glob.glob(save_gif + "/iter/*.png"), key=os.path.getmtime):
        images.append(Image.open(file))
        if double_init : images.append(Image.open(file)); double_init = False

    images[0].save(fp=save_gif + "/" + str(name) + date.strftime("_%d%b%H%M") +".gif", format='GIF', append_images=images, save_all=True, duration=500,
                   loop=0)

    ii = 0
    for file in sorted(glob.glob(save_gif + "/iter/*.png")):
        images[ii].close()
        try:
            os.remove(file)
        except PermissionError:
            print(
                "Can't delete saves for gif. A process might not have closed properly. "
                "\nIgnore deletion \nI highly suggest you to delete it.")
        ii += 1
    os.rmdir(save_gif + "/iter/")