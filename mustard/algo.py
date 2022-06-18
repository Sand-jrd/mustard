#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:27:42 2021

______________________________

| algorithms used in neo-mayo |
______________________________

@author: sand-jrd
"""

from vip_hci.pca import pca_fullfr, pca_annular
from vip_hci.preproc import cube_derotate
from vip_hci.preproc import frame_rotate
from vip_hci.itpca import pca_it
from vip_hci.var import frame_center

import torch
from torch.nn.functional import conv2d
import torch.fft as tf

import numpy as np


# %% Initialisation / kind of PCA / PCA iter

def init_estimate(cube: np.ndarray, angle_list: np.ndarray, Imode='max_common', **kwarg) -> (np.ndarray, np.ndarray):
    """
    Estimate Satrlight and Circoncstelar light using pca

    Parameters
    ----------
    cube : np.ndarray
        Cube of ADI science data

    angle_list : np.ndarray
        Rotation angle associated with the ADI cube

    Imode : {'pca',''pcait}
        mode of pca

    kwarg :
        Arguments that will be pass to vip function for pca or pcait

    Returns
    -------
    L and X : (np.ndarray, np.ndarray)
        Estimated starlight and circonstelar light
    """

    L_k = np.zeros(cube.shape)
    nb_frame = cube.shape[0]

    if Imode == "pca":
        print("Mode pca")
        res = pca_fullfr.pca(cube, angle_list, verbose=False, **kwarg)

        for frame_id in range(nb_frame):
            frame_id_rot = frame_rotate(res, angle_list[frame_id])
            L_k[frame_id] = cube[frame_id] - (frame_id_rot.clip(min=0))

    elif Imode == "pcait":
        print("Mode pca iterative")
        res = pca_it(cube, angle_list, verbose=False, **kwarg)

        for frame_id in range(nb_frame):
            frame_id_rot = frame_rotate(res, angle_list[frame_id])
            L_k[frame_id] = cube[frame_id] - (frame_id_rot.clip(min=0))

    elif Imode == "pca_annular":
        print("Mode pca annular")
        res = pca_annular(cube, angle_list, verbose=False, **kwarg)

        for frame_id in range(nb_frame):
            frame_id_rot = frame_rotate(res, angle_list[frame_id])
            L_k[frame_id] = cube[frame_id] - (frame_id_rot.clip(min=0))

    elif Imode == "max_common":
        print("Mode maximum in common")

        science_data_derot = cube_derotate(cube, list(angle_list))
        science_data_derot = science_data_derot

        res = np.min(science_data_derot, 0)

        for frame_id in range(nb_frame):
            frame_id_rot = frame_rotate(res, angle_list[frame_id])
            L_k[frame_id] = cube[frame_id] - (frame_id_rot.clip(min=0))

    else : raise ValueError(str(Imode) + " is not a valid mode to init estimator.\nPossible values are {'max_common',"
                                         "'pca_annular','pca','pcait'}")

    return  np.median(L_k, axis=0).clip(min=0), res.clip(min=0)


# %% Operator on tensors
# Mostly copies of vip functiun adapted to tensors

def laplacian_tensor_conv(tensor: torch.Tensor, kernel_size=3) -> torch.Tensor:
    """
    Apply laplacian filter on input tensor X

    Parameters
    ----------
    tensor : torch.Tensor
        input tensor
    kernel_size : {3, 5 , 7}
        Lpalacian kernel size

    Returns
    -------
    torch.Tensor

    """

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


def sobel_tensor_conv(tensor: torch.Tensor, axis="y") -> torch.Tensor:
    """
    Apply 3x3 sobel filter on input tensor X

    Parameters
    ----------
    tensor : torch.tensor
        input tensor

    axis : {'y','x'}
        direction of the sobel filter

    Returns
    -------
    torch.Tensor

    """

    if axis == "y":
        kernel = np.array([[[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]], dtype='float64')
    elif axis == "x":
        kernel = np.array([[[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]], dtype='float64')
    else : raise ValueError("'Axis' parameters should be 'x' or 'y', not"+str(axis))

    kernel = torch.unsqueeze(torch.from_numpy(kernel), 0).double()

    shape = tensor.shape
    filtered = conv2d(tensor.reshape( (1,) + shape), kernel, padding='same')

    return filtered


def gaussian_tensor_conv(tensor: torch.Tensor, k_size = 5) -> torch.Tensor:
    """
    Apply 3x3 gaussian filter on input tensor X

    Parameters
    ----------
    tensor : torch.tensor
        input tensor

    k_size : int
        kenrel size

    Returns
    -------
    torch.Tensor

    """

    if k_size == 3 :
        kernel = np.array([[[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]], dtype='float64')
    elif k_size == 5:
        kernel = np.array([[[1,  4,  6,  4, 1],
                            [4, 18, 30, 18, 4],
                            [6, 30, 48, 30, 6],
                            [4, 18, 30, 18, 4],
                            [1,  4,  6,  4, 1]]], dtype='float64')
    else : raise(ValueError("Kernel size can be {3,5}"))

    kernel = torch.unsqueeze(torch.from_numpy(kernel), 0).double()
    filtered = conv2d(tensor, kernel, padding='same')

    return filtered


def tensor_rotate_fft(tensor: torch.Tensor, angle: float) -> torch.Tensor:
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
    tensor : torch.Tensor
        Input image, 2d array.
    angle : float
        Rotation angle.

    Returns
    -------
    array_out : torch.Tensor
        Resulting frame.

    """
    y_ori, x_ori = tensor.shape[1:]

    while angle < 0:
        angle += 360
    while angle > 360:
        angle -= 360

    if angle > 45:
        dangle = angle % 90
        if dangle > 45:
            dangle = -(90 - dangle)
        nangle = int(np.rint(angle / 90))
        tensor_in = torch.rot90(tensor, nangle, [1, 2])
    else:
        dangle = angle
        tensor_in = tensor.clone()

    if y_ori%2 or x_ori%2:
        # NO NEED TO SHIFT BY 0.5px: FFT assumes rot. center on cx+0.5, cy+0.5!
        tensor_in = tensor_in[:, :-1, :-1]

    a = np.tan(np.deg2rad(dangle) / 2).item()
    b = -np.sin(np.deg2rad(dangle)).item()

    y_new, x_new = tensor_in.shape[1:]
    arr_xy = torch.from_numpy(np.mgrid[0:y_new, 0:x_new])
    cy, cx = frame_center(tensor[0])
    arr_y = arr_xy[0] - cy
    arr_x = arr_xy[1] - cx

    s_x = tensor_fft_shear(tensor_in, arr_x, a, ax=2)
    s_xy = tensor_fft_shear(s_x, arr_y, b, ax=1)
    s_xyx = tensor_fft_shear(s_xy, arr_x, a, ax=2)

    if y_ori % 2 or x_ori % 2:
        # set it back to original dimensions
        array_out = torch.zeros([1, s_xyx.shape[1]+1, s_xyx.shape[2]+1])
        array_out[0, :-1, :-1] = torch.real(s_xyx)
    else:
        array_out = torch.zeros([1, s_xyx.shape[1], s_xyx.shape[2]])
        array_out = torch.real(s_xyx)

    return array_out


def tensor_fft_shear(arr, arr_ori, c, ax):
    ax2 = 1 - (ax-1) % 2
    freqs = tf.fftfreq(arr_ori.shape[ax2], dtype=torch.float64)
    sh_freqs = tf.fftshift(freqs)
    arr_u = torch.tile(sh_freqs, (arr_ori.shape[ax-1], 1))
    if ax == 2:
        arr_u = torch.transpose(arr_u, 0, 1)
    s_x = tf.fftshift(arr)
    s_x = tf.fft(s_x, dim=ax)
    s_x = tf.fftshift(s_x)
    s_x = torch.exp(-2j * torch.pi * c * arr_u * arr_ori) * s_x
    s_x = tf.fftshift(s_x)
    s_x = tf.ifft(s_x, dim=ax)
    s_x = tf.fftshift(s_x)

    return s_x


def tensor_conv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(torch.fft.fft2(x)) *
                                                         torch.fft.fftshift(torch.fft.fft2(y)))))
