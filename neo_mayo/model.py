#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:42:10 2021

______________________________

| Forward model used in mayo |
______________________________

@author: sand-jrd
"""
import numpy
import numpy as np
import torch
from neo_mayo.algo import tensor_rotate_fft
from torchvision.transforms.functional import rotate, InterpolationMode

import warnings

warnings.simplefilter("ignore", UserWarning)


# %% Forward ADI model :

class model_ADI:
    """ Forward models as presented in mayo
        
            Y = L + R(x)
        
        with L = starlight/constant flux in the image
             x = circumstellar flux
     """

    def __init__(self, rot_angles: np.array, coro: np.array, psf: None or np.array, rot="fft", weighted_rot=True):

        # -- Constants
        # Sizes
        self.nb_frame = len(rot_angles)
        self.frame_shape = coro.shape

        # Frames known rotations list
        rot_angles = normlizangle(rot_angles)
        self.rot_angles = rot_angles

        # Compute ponderation weight by rotations
        self.ang_weight = compute_rot_weight(rot_angles) if weighted_rot else np.ones(self.nb_frame)

        # Check if deconvolution mode (and pad psf if needed)
        if psf is not None:
            if psf.shape != coro.shape: psf = pad_psf(psf, coro.shape)
            self.psf = torch.unsqueeze(torch.from_numpy(psf), 0)
        else: self.psf = None

        # Coro mask
        self.coro = torch.unsqueeze(torch.from_numpy(coro), 0)

        # -- Functions

        if rot == "fft":
            self.rot = tensor_rotate_fft
            self.rot_args = {}

        else: # TODO :  remove
            self.rot = rotate  # Torchvison rotation
            self.rot_args = {'interpolation': InterpolationMode.BILINEAR}

        # TODO :  fction in utils
        self.conv = lambda x, y, **kwargs: torch.abs(torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.fftshift(torch.fft.fft2(x)) * torch.fft.fftshift(torch.fft.fft2(y)))))

    def forward_ADI(self, L: torch.Tensor, x: torch.Tensor, flux=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * L + R(x)) )  """

        if flux is None: flux = torch.ones(self.nb_frame - 1)

        Y = torch.zeros((self.nb_frame,) + L.shape).double()

        # First image. No intensity vector
        Rx = self.rot(x.abs(), float(self.rot_angles[0]), **self.rot_args)
        Y[0] = self.ang_weight[0] * (L + self.coro * Rx)

        for frame_id in range(1, self.nb_frame):
            Rx = self.rot(x.abs(), float(self.rot_angles[frame_id]), **self.rot_args)
            if self.psf is not None: Rx = self.conv(Rx, self.psf)
            Y[frame_id] = self.ang_weight[frame_id] * (flux[frame_id - 1] * L + self.coro * Rx)

        return Y


def normlizangle(angles: numpy.array) -> numpy.array:
    """Normaliz an angle between 0 and 360"""

    angles[angles < 0] += 360
    angles = angles % 360

    return angles

def normalize(v: numpy.array) -> numpy.array:
    """Normaliz an vector"""

    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def compute_rot_weight(angs: numpy.array) -> numpy.array:
    nb_frm = len(angs)
    max_rot = 1
    ang_weight = []
    for id_ang, ang in enumerate(angs) :
        nb_neighbours = 0
        detla_ang = 0; tmp_id = id_ang

        while detla_ang < max_rot and tmp_id<nb_frm-1:  # neighbours after
            tmp_id+=1; nb_neighbours+=1
            detla_ang+=abs(angs[tmp_id] - angs[id_ang])
        w_1 = (1+abs(angs[id_ang] - angs[id_ang + 1])) / nb_neighbours \
            if nb_neighbours > 1 else 1

        tmp_id = id_ang; detla_ang = 0; nb_neighbours = 0
        while detla_ang < max_rot and tmp_id>0:  # neighbours before
            tmp_id -= 1;nb_neighbours += 1
            detla_ang += abs(angs[tmp_id] - angs[id_ang])
        w_2= (1+abs(angs[id_ang] - angs[id_ang-1])) / nb_neighbours \
            if nb_neighbours > 1 else 1

        ang_weight.append((w_2+w_1)/2)

    return np.array(ang_weight)

def pad_psf(M: numpy.array, shape)-> numpy.array:
    """Pad with 0 the PSF if it is too small """
    if M.shape[0] % 2:
        raise ValueError("PSF shape should be wise : psf must be centred on 4pix not one")
    M_resized = np.zeros(shape)
    mid = shape[0] // 2
    psfmid = M.shape[0] // 2
    M_resized[mid - psfmid:mid + psfmid, mid - psfmid:mid + psfmid] = M

    return M_resized