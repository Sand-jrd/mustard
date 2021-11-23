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

# Syntax wrapper for simplify minimize  usage.
import torch
from torch.nn.functional import conv2d
import torch.fft as tf

# File management
from vip_hci.fits import open_fits
import json, glob, os


def var_inline(L, x):
    """ Wrap parameters as a vector to fit minimize syntax  Parameters
    
    Parameters 
    ----------
     L,x : ndarray
         Variables input of minimize

     Returns
     -------
     x0 : array
         inline array of the concatenated variables
     
  """
    if not (len(L.shape) == 2 and len(x.shape) == 2 and x.shape == L.shape):
        raise ValueError('L and x are supposed to be matrix 2D of same dimensions')
    return np.concatenate((L.flatten(), x.flatten()), axis=None)


def var_inmatrix(M, size):
    """ Unwrap parameters from minimize (which is a big vector) into matrices
        Matching the order we wrapped it in var inline.
   
    Parameters
    ----------
    M : array
        Variables taken from minimize which is an inline array of
        concatenated parameters
    
    size : int
        dimensions of matrix
    
    Returns
    -------
    L,x : ndarray
        Unwrapped parameters
    
 """
    if M.size != (2 * size * size):
        raise ValueError("length of vector doesn't match expected value")

    mat_length = size * size

    # As we define in var_inline, L should be first then x
    L = M[:mat_length].reshape(size, size)
    x = M[mat_length:].reshape(size, size)

    return L, x


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


def laplacian_tensor_conv(tensor, kernel_size=3):
    """ Apply laplacian filter on input tensor X"""

    kernel3 = torch.Tensor([[[[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]]]])
    kernel5 = torch.Tensor([[[[-4, -1, 0, -1, -4],
                              [-1, 2, 3, 2, -1],
                              [0, 3, 4, 3, 0],
                              [-1, 2, 3, 2, -1],
                              [-4, -1, 0, -1, -4]]]])
    kernel7 = torch.Tensor([[[[-10, -5, -2, -1, -2, -5, -10],
                              [-5, 0, 3, 4, 3, 0, -5],
                              [-2, 3, 6, 7, 6, 3, -2],
                              [-1, 4, 7, 8, 7, 4, -1],
                              [-2, 3, 6, 7, 6, 3, -2],
                              [-5, 0, 3, 4, 3, 0, -5],
                              [-10, -5, -2, -1, -2, -5, -10]]]])
    if kernel_size == 3:
        kernel = kernel3
    elif kernel_size == 5:
        kernel = kernel5
    elif kernel_size == 7:
        kernel = kernel7
    else:
        raise ValueError('Kernel size must be either 3, 5 or 7.')
    filtered = conv2d(torch.unsqueeze(tensor, 0), kernel, padding='same')

    return filtered


def sobel_tensor_conv(tensor):
    """ Apply laplacian filter on input tensor X"""

    kernel = torch.Tensor([[[[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]]])
    filtered = conv2d(torch.unsqueeze(tensor, 0), kernel, padding='same')

    return filtered


save_gif = "./"


def tensor_rotate_fft(tensor_in:torch.Tensor, angle:float) -> torch.Tensor:
    """ Rotates Tensor using Fourier transform phases:
        Rotation = 3 consecutive lin. shears = 3 consecutive FFT phase shifts
        See details in Larkin et al. (1997) and Hagelberg et al. (2016).
        Note: this is significantly slower than interpolation methods
        (e.g. opencv/lanczos4 or ndimage), but preserves the flux better
        (by construction it preserves the total power). It is more prone to
        large-scale Gibbs artefacts, so make sure no sharp edge is present in
        the image to be rotated.

        /!\ This is a blindly coded adaptation for Tensor of the vip function rotate_fft
        (https://github.com/vortex-exoplanet/VIP/blob/51e1d734dcdbee1fbd0175aa3d0ab62eec83d5fa/vip_hci/preproc/derotation.py#L507)

        /!\ This suppose the frame is perfectly centred

        ! Warning: if input frame has even dimensions, the center of rotation
        will NOT be between the 4 central pixels, instead it will be on the top
        right of those 4 pixels. Make sure your images are centered with
        respect to that pixel before rotation.

    Parameters
    ----------
    tensor_in : torch.Tensor
        Input image, 2d array.
    angle : float
        Rotation angle.

    Returns
    -------
    array_out : torch.Tensor
        Resulting frame.

    """
    y_ori, x_ori = tensor_in.shape[1:]

    while angle < 0:
        angle += 360
    while angle > 360:
        angle -= 360

    if angle > 45:
        dangle = angle % 90
        if dangle > 45:
            dangle = -(90 - dangle)
        nangle = int(np.rint(angle / 90))
        tensor_in = torch.rot90(tensor_in, nangle, [1, 2])
    else:
        dangle = angle

    if y_ori % 2 or x_ori % 2:
        # NO NEED TO SHIFT BY 0.5px: FFT assumes rot. center on cx+0.5, cy+0.5!
        tensor_in = tensor_in[0, :-1, :-1]

    a = np.tan(np.deg2rad(dangle) / 2).item()
    b = -np.sin(np.deg2rad(dangle)).item()

    arr_xy = torch.from_numpy(np.mgrid[0:y_ori, 0:x_ori])

    s_x   = tensor_fft_shear(tensor_in, arr_xy[0], a, ax=2)
    s_xy  = tensor_fft_shear(s_x, arr_xy[1], b, ax=1)
    s_xyx = tensor_fft_shear(s_xy, arr_xy[0], a, ax=2)

    if y_ori % 2 or x_ori % 2:
        # shift + crop back to odd dimensions , using FFT
        array_out = torch.zeros([s_xyx.shape[0] + 1, s_xyx.shape[1] + 1])
        # NO NEED TO SHIFT BY 0.5px: FFT assumes rot. center on cx+0.5, cy+0.5!
        array_out[:-1, :-1] = torch.real(s_xyx)
    else:
        array_out = torch.real(s_xyx)

    return array_out


def tensor_fft_shear(arr, arr_ori, c, ax, pad=0, shift_ini=True):
    ax2 = 1 - (ax-1) % 2
    freqs = tf.fftfreq(arr_ori.shape[ax2])
    sh_freqs = tf.fftshift(freqs)
    arr_u = torch.tile(sh_freqs, (arr_ori.shape[ax-1], 1))
    if ax == 2:
        arr_u = arr_u.T
    s_x = tf.fftshift(arr)
    s_x = tf.fft(s_x, axis=ax)
    s_x = tf.fftshift(s_x)
    s_x = np.exp(-2j * torch.pi * c * arr_u * arr_ori) * s_x
    s_x = tf.fftshift(s_x)
    s_x = tf.ifft(s_x, axis=ax)
    s_x = tf.fftshift(s_x)

    return s_x


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


def iter_to_gif(name="sim"):
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
