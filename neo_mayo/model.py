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

    def __init__(self, rot_angles: np.array, coro: np.array, psf: None or np.array, rot="fft"):

        # -- Constants
        # Sizes
        self.nb_frame = len(rot_angles)
        self.frame_shape = coro.shape

        # Frames known rotations list
        self.rot_angles = rot_angles

        # Check if deconvolution mode (and pad psf if needed)
        if psf is not None:
            if psf.shape != coro.shape: psf = pad_psf(psf, coro.shape)
            self.psf = torch.unsqueeze(torch.from_numpy(psf), 0)
        else:
            self.psf = None

        # Coro mask
        self.coro = torch.unsqueeze(torch.from_numpy(coro), 0)

        # -- Functions

        if rot == "fft":
            self.rot = tensor_rotate_fft
            self.rot_args = {}

        else:  # TODO :  remove
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
        Y[0] = L + self.coro * Rx

        for frame_id in range(1, self.nb_frame):
            Rx = self.rot(x.abs(), float(self.rot_angles[frame_id]), **self.rot_args)
            if self.psf is not None: Rx = self.conv(Rx, self.psf)
            Y[frame_id] = flux[frame_id - 1] * L + self.coro * Rx

        return Y



def pad_psf(M: numpy.array, shape) -> numpy.array:
    """Pad with 0 the PSF if it is too small """
    if M.shape[0] % 2:
        raise ValueError("PSF shape should be wise : psf must be centred on 4pix not one")
    M_resized = np.zeros(shape)
    mid = shape[0] // 2
    psfmid = M.shape[0] // 2
    M_resized[mid - psfmid:mid + psfmid, mid - psfmid:mid + psfmid] = M

    return M_resized
