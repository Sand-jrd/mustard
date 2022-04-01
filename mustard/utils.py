#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:37:16 2021

______________________________

| Utils Function for mustard  |
______________________________


@author: sand-jrd
"""

# Misc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from PIL import Image
import numpy as np
import torch

# File management
from vip_hci.fits import open_fits
import json, glob, os
from os import mkdir
from os.path import isdir
from datetime import datetime

BoxStyle  = {"boxstyle": 'square', "pad": 1.3, "mutation_scale": 0.01, "facecolor": 'indigo', "alpha": 0.65}
FontStyle = {"family": 'sans', "color": 'beige', "weight": 'normal', "size": 12}
estimBox  = {"boxstyle": 'square', "facecolor": 'beige', "edgecolor": 'indigo', "alpha": 0.65}
titleFont = {'color': 'indigo', "weight": "bold", "backgroundcolor": 'beige'}


# %% Create patterns

def circle(shape: tuple, r: float, offset=(0.5, 0.5)):
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
    assert(len(shape) == 2 or len(shape) == 3)
    if isinstance(offset, (int, float)): offset = (offset, offset)

    nb_f  = shape[0]  if len(shape) == 3 else 0
    shape = shape[1:] if len(shape) == 3 else shape

    M = np.zeros(shape)
    w, l = shape
    for x in range(0, w):
        for y in range(0, l):
            if pow(x - (w / 2) + offset[0], 2) + pow(y - (l / 2) + offset[1], 2) < pow(r, 2):
                M[x, y] = 1

    if nb_f: M = np.tile(M, (nb_f, 1, 1))

    return M


def ellipse(shape: tuple, small_ax: float, big_ax: float, rotation: float, off_center=(0, 0)) -> np.ndarray:
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

    nb_f  = shape[0]  if len(shape) == 3 else 0
    shape = shape[1:] if len(shape) == 3 else shape

    mid =  np.array(shape) // 2 - np.array(off_center)
    M = np.zeros(shape)
    rotation = np.deg2rad(rotation)

    def isInEllipse(xc: float, yc: float) -> bool:
        """Tell if the point (x,y) is inside the ellipse"""
        term1 = (xc - mid[0])*np.cos(rotation) +  (yc - mid[1])*np.sin(rotation)
        term2 = (xc - mid[0])*np.sin(rotation) - (yc - mid[1])*np.cos(rotation)
        return (term1 / small_ax)**2 + (term2 / big_ax)**2  <= 1

    # Generate elips coordinate trace points
    for x in range(shape[0]):
        for y in range(shape[1]):
            if isInEllipse(x, y):
                M[x, y] = 1

    if nb_f : M = np.tile(M, (nb_f, 1, 1))

    return M


def gaussian(shape, sigma = 1, mu = 0) :
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]))
    dst = np.sqrt(x * x + y * y)
    return np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))


# %% Manage files
def unpack_science_datadir(datadir: str) -> (torch.Tensor, torch.Tensor):
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
    ispsf = 'psf' in data_info.keys()

    # Open fits
    angles = open_fits(datadir + "/" + data_info["angles"], verbose=False)
    science_data = open_fits(datadir + "/" + data_info["cube"], verbose=False)
    psf = open_fits(datadir + "/" + data_info["psf"], verbose=False) if ispsf else None

    return angles, science_data, psf


def print_iter(L: torch.Tensor, x: torch.Tensor, flux: torch.Tensor, bfgs_iter: int, msg_box: str,
               extra_msg: str or None, datadir: str or bool, coro : torch.Tensor(1)) -> None:

    L_np  = abs(L.detach().numpy()[0, :, :])
    X_np  = abs(x.detach().numpy()[0, :, :])
    coro  = coro.numpy()
    fluxnp = flux.detach().numpy()

    plt.ioff()
    col = 3 if flux.requires_grad else 2
    ratios = [5, 1, 2] if flux.requires_grad else [3, 1]

    plt.subplots(col, 2, figsize=(16, 9), gridspec_kw={'height_ratios': ratios})

    plt.suptitle("Iteration n°" + str(bfgs_iter))
    vmin = np.percentile(L_np, 0) if np.percentile(L_np, 0) > 0 else np.percentile(X_np, 80)
    vmin = 1e-10 if vmin == 0 else vmin
    args = {"cmap": "gnuplot2", "norm": LogNorm(vmax=np.percentile(coro*L_np, 100), vmin=vmin)}

    if not bfgs_iter : title = "Initialisation"
    else : title = "Estimation of L and X"
    if extra_msg : title += "\n" + extra_msg; titleFont["color"] = 'red'
    if flux.requires_grad : title += " and flux"
    title += " at iteration n°" + str(bfgs_iter)

    plt.suptitle(title, **titleFont)

    plt.subplot(col, 2, 1), plt.imshow(coro * np.abs(L_np), **args)
    plt.title("L (starlight) "), plt.colorbar()
    plt.subplot(col, 2, 2), plt.imshow(coro * np.abs(X_np), **args)
    plt.title("X (circonstellar light)"), plt.colorbar()

    if flux.requires_grad :

        plt.subplot2grid((col, 2), (1, 0), colspan=2)
        plt.bar(range(2, len(fluxnp)+2), fluxnp-1, color='beige', edgecolor="black")
        plt.xticks(range(1, len(fluxnp)+2))
        plt.ylabel("I factor diff"), plt.xlabel("frameID"), plt.title("flux variations")

        LIinfo = "I min : " + str(np.min(fluxnp)) + "\nI max : " + str(np.max(fluxnp))
        plt.text(0, 0, LIinfo, bbox=estimBox, fontdict={'color': 'indigo'})

    plt.subplot2grid((col, 2), (2 if flux.requires_grad else 1, 0), colspan=2)
    plt.axis('off'), plt.text(0.3, -0.2, msg_box.expandtabs(), bbox=BoxStyle, fontdict=FontStyle)

    if not datadir : datadir = "."
    if not isdir(datadir + "/iter/"): mkdir(datadir + "/iter/")

    plt.savefig(datadir + "/iter/" + str(bfgs_iter) + ".png", pad_inches=0.5)
    plt.clf(), plt.close()


def iter_to_gif(save_gif='./', suffix=None) -> None:
    images = []
    plt.ion()

    date = datetime.now()
    if not save_gif: save_gif = "./"
    suffix = '' if not suffix else suffix + '_'

    double_init = True
    for file in sorted(glob.glob(save_gif + "/iter/*.png"), key=os.path.getmtime):
        images.append(Image.open(file))
        if double_init : images.append(Image.open(file)); double_init = False

    images[0].save(fp=save_gif + "/sim_" + str(suffix) + date.strftime("_%d%b%H%M") + ".gif", format='GIF',
                   append_images = images, save_all=True, duration=500, loop=0)

    ii = 0
    for file in sorted(glob.glob(save_gif + "/iter/*.png")):
        images[ii].close()
        try: os.remove(file)
        except Exception as e: print("[WARNING] Failed to delete iter .png : " + str(e))
        ii += 1

    try: os.rmdir(save_gif + "/iter/")
    except Exception as e : print("[WARNING] Failed to remove iter dir : " + str(e))
