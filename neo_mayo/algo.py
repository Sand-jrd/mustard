#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:27:42 2021

______________________________

| algorithms used in neo-mayo |
______________________________

@author: sand-jrd
"""

from vip_hci.pca import pca_fullfr
from vip_hci.preproc import frame_rotate
from vip_hci.itpca import pca_it

import torch
from torch.nn.functional import conv2d
import torch.fft as tf

import numpy as np


# %% Initialisation / kind of PCA / PCA iter

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


# %% Operator on tensors
# Mostly copies of vip functiun adapted to tensors

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
    """ Apply sovel filter on input tensor X"""

    kernel = np.array([[[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]], dtype='float64')

    kernel = torch.unsqueeze(torch.from_numpy(kernel), 0).double()

    shape = tensor.shape
    filtered = conv2d(tensor.reshape( (1,) + shape), kernel, padding='same')

    return filtered

def tensor_rotate_fft(tensor_in: torch.Tensor, angle: float) -> torch.Tensor:
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

    y_ori, x_ori = tensor_in.shape[1:]

    a = np.tan(np.deg2rad(dangle) / 2).item()
    b = -np.sin(np.deg2rad(dangle)).item()

    arr_xy = torch.from_numpy(np.mgrid[0:y_ori, 0:x_ori])
    arr_xy -= x_ori // 2

    s_x = tensor_fft_shear(tensor_in, arr_xy[1], a, ax=2)
    s_xy = tensor_fft_shear(s_x, arr_xy[0], b, ax=1)
    s_xyx = tensor_fft_shear(s_xy, arr_xy[1], a, ax=2)

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
    freqs = tf.fftfreq(arr_ori.shape[ax2], dtype=torch.float64)
    sh_freqs = tf.fftshift(freqs)
    arr_u = torch.tile(sh_freqs, (arr_ori.shape[ax-1], 1))
    if ax == 2:
        arr_u = torch.transpose(arr_u,0,1)
    s_x = tf.fftshift(arr)
    s_x = tf.fft(s_x, dim=ax)
    s_x = tf.fftshift(s_x)
    s_x = torch.exp(-2j * torch.pi * c * arr_u * arr_ori) * s_x
    s_x = tf.fftshift(s_x)
    s_x = tf.ifft(s_x, dim=ax)
    s_x = tf.fftshift(s_x)

    return s_x
