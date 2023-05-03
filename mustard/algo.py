#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:27:42 2021

______________________________

| algorithms used in neo-mayo |
______________________________

@author: sand-jrd
"""

from vip_hci.var import frame_center

import torch
from torch.nn.functional import conv2d
import torch.fft as tf

import numpy as np
from skimage.filters import threshold_multiotsu
from vip_hci.var import frame_filter_lowpass

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if axis == "y":
        kernel = np.array([[[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]], dtype='float64')
    elif axis == "x":
        kernel = np.array([[[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]], dtype='float64')
    else : raise ValueError("'Axis' parameters should be 'x' or 'y', not"+str(axis))

    kernel = torch.unsqueeze(torch.from_numpy(kernel), 0).double().to(device)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    kernel = torch.unsqueeze(torch.from_numpy(kernel), 0).double().to(device)
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

        (!) This is a blindly coded adaptation for Tensor of the vip function rotate_fft
        (https://github.com/vortex-exoplanet/VIP/blob/51e1d734dcdbee1fbd0175aa3d0ab62eec83d5fa/vip_hci/preproc/derotation.py#L507)

        (!) This suppose the frame is perfectly centred

        (!) Warning: if input frame has even dimensions, the center of rotation
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

    if y_ori % 2 or x_ori % 2:
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


def tensor_fft_scale(array: torch.Tensor, scale: float, ori_dim=True):
    """
    Resample the frames of a cube with a single scale factor using a FFT-based
    method.
    Parameters
    ----------
    array : 3d tensor
        Input cube, 3d array.
    scale : int or float
        Scale factor for upsampling or downsampling the frames in the cube. If
        a tuple it corresponds to the scale along x and y.
    ori_dim: bool, opt
        Whether to crop/pad scaled array in order to have the output with the
        same dimensions as the input array. By default, the x,y dimensions of
        the output are the closest integer to scale*dim_input, with the same
        parity as the input.
    Returns
    -------
    array_resc : numpy ndarray
        Output cube with resampled frames.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if scale == 1:
        return array



    if array.shape[0] % 2 :
        odd = True
        array_even = torch.zeros([array.shape[1] + 1, array.shape[2] + 1])
        array_even[1:, 1:] = array[0]
        array = array_even
    else:
        array_even = torch.zeros([array.shape[1], array.shape[2]])
        array_even[:, :] = array[0]
        array = array_even
        odd = False

    dim = array.shape[0]  # even square
    kd_array = torch.arange(dim//2 + 1)

    # scaling factor chosen as *close* as possible to N''/N', where:
    #   N' = N + 2*KD (N': dim after FT)
    #   N" = N + 2*KF (N'': dim after FT-1 of FT image),
    #   => N" = 2*round(N'*sc/2)
    #   => KF = (N"-N)/2 = round(N'*sc/2 - N/2)
    #         = round(N/2*(sc-1) + KD*sc)
    # We call yy=N/2*(sc-1) +KD*sc
    yy = dim/2 * (scale - 1) + kd_array.double().to(device) * scale

    # We minimize the difference between the `ideal' N" and its closest
    # integer value by minimizing |yy-int(yy)|.
    kf_array = torch.round(yy).int()
    tmp = torch.abs(yy-kf_array)
    imin = torch.argmin(tmp)  # Nan values not handled

    kd_io = kd_array[imin]
    kf_io = kf_array[imin]

    # Extract a part of array and place into dim_p array
    dim_p = int(dim + 2*kd_io)
    tmp = torch.zeros((dim_p, dim_p)).double().to(device)
    tmp[kd_io:kd_io+dim, kd_io:kd_io+dim] = array

    # Fourier-transform the larger array
    array_f = tf.fftshift(tf.fft2(tmp))

    # Extract a part of, or expand, the FT to dim_pp pixels
    dim_pp = int(dim + 2*kf_io)

    if dim_pp > dim_p:
        tmp = torch.zeros((dim_pp, dim_pp), dtype=torch.cfloat)
        tmp[(dim_pp-dim_p)//2:(dim_pp+dim_p)//2,
            (dim_pp-dim_p)//2:(dim_pp+dim_p)//2] = array_f
    else:
        tmp = array_f[kd_io-kf_io:kd_io-kf_io+dim_pp,
                      kd_io-kf_io:kd_io-kf_io+dim_pp]

    # inverse Fourier-transform the FT
    tmp = tf.ifft2(tf.fftshift(tmp))
    array_resc = torch.real(tmp)
    del tmp

    # Extract a part of or expand the scaled image to desired number of pixels
    dim_resc = int(round(scale*dim))
    if dim_resc > dim and dim_resc % 2 != dim % 2:
        dim_resc += 1
    elif dim_resc < dim and dim_resc % 2 != dim % 2:
        dim_resc -= 1  # for reversibility

    if not ori_dim and dim_pp > dim_resc:
        array_resc = array_resc[(dim_pp-dim_resc)//2:(dim_pp+dim_resc)//2,
                                (dim_pp-dim_resc)//2:(dim_pp+dim_resc)//2]
    elif not ori_dim and dim_pp <= dim_resc:
        array = torch.zeros((dim_resc, dim_resc)).double().to(device)
        array[(dim_resc-dim_pp)//2:(dim_resc+dim_pp)//2,
              (dim_resc-dim_pp)//2:(dim_resc+dim_pp)//2] = array_resc
        array_resc = array
    elif dim_pp > dim:
        array_resc = array_resc[kf_io:kf_io+dim, kf_io:kf_io+dim]
    elif dim_pp <= dim:
        scaled = array*0
        scaled[-kf_io:-kf_io+dim_pp, -kf_io:-kf_io+dim_pp] = array_resc
        array_resc = scaled

    # array_resc /= scale * scale

    if odd:
        array_tmp = torch.zeros([1, array_resc.shape[0]-1, array_resc.shape[1]-1])
        array_tmp[0] = array_resc[1:, 1:]
        array_resc = array_tmp
    else :
        array_tmp = torch.zeros([1, array_resc.shape[0], array_resc.shape[1]])
        array_tmp[0] = array_resc
        array_resc = array_tmp

    return array_resc


def tensor_conv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.abs(tf.ifftshift(tf.ifft2(tf.fftshift(tf.fft2(x)) * tf.fftshift(tf.fft2(y)))))


def convert_to_mask(img: np.ndarray):
    """ Convert an image into a binary mask

    Parameters
    ----------
    img : np.ndarray

    Returns
    -------
    mask : np.ndarray

    """

    thresholds = threshold_multiotsu(img)
    val = thresholds[0]
    img_m = 1*(img < val)
    mask = frame_filter_lowpass(img_m)

    return mask

# %% Radial profile

def create_radial_prof_matirx(shape:torch.Tensor, bin_size=1, r2_scale=False) -> torch.Tensor:

    y, x = np.indices(shape)
    y, x = torch.from_numpy(y), torch.from_numpy(x)

    mid = shape[0] / 2
    rr = torch.sqrt((x - mid) ** 2 + (y - mid) ** 2)

    rad_max = torch.max(rr)

    nb_anns = torch.floor(rad_max / bin_size).type(torch.IntTensor)
    rad_prof_transform = torch.ones(shape[0]**2, nb_anns, dtype=torch.double)

    for r in range(nb_anns):
        ann = (rr < (r * bin_size) + 1) * (rr >= r * bin_size)
        rad_prof_transform[:, r] *= torch.flatten(ann)/torch.sum(ann)
        if r2_scale : rad_prof_transform[:, r] *= r**2

    return rad_prof_transform

def radial_profil(M: torch.Tensor, rad_prof_transform: torch.Tensor, norm_bkg=10):
    """Generate radial profil using transformation matrix. 
    Use 'create_radial_prof_matirx' to generate rad_prof_transform"""
    
    radial_prof = torch.flatten(M) @ rad_prof_transform
    if norm_bkg : radial_prof -= torch.mean(radial_prof[-norm_bkg:])

    return radial_prof/torch.max(radial_prof)

def res_non_convexe(Rp: torch.Tensor, pup_size=8) -> float :
    """Sum of positive slope -- can be used as a regualrizator to penalize
    non-negative slope and enforce a slope that """
    
    slope = (Rp[pup_size + 1:] - Rp[pup_size:-1]).clip(min=0)

    return torch.sum(slope)

def radial_profil_bins(M: torch.Tensor, y, x, bin_size=1, norm_bkg=10) -> torch.Tensor:
    """Generate radial profil. Same operation as 'radial_profil' but do not generate 
    the tranformation matrix. Not optimal for a ML usage"""
    
    mid = M.shape[0] // 2
    rr = torch.sqrt((x - mid) ** 2 + (y - mid) ** 2)

    means = []
    rad_max = torch.max(rr)

    nb_anns = torch.floor(rad_max / bin_size).type(torch.IntTensor)
    for r in range(nb_anns):
        ann = (rr < (r * bin_size) + 1) * (rr >= r * bin_size)
        means.append(torch.mean(M[ann]))

    radial_prof = torch.FloatTensor(means)
    if norm_bkg : radial_prof -= torch.mean(radial_prof[-norm_bkg:])

    return radial_prof
