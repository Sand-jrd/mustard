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
from neo_mayo.algo import tensor_rotate_fft, tensor_conv
from torch.nn import ReLU as reluConstr
ReLU = reluConstr()


# %% Forward ADI model :
class model_ADI:
    """ Forward models as presented in mayo
        
            Y = L + R(x)
        
        with L = starlight/constant flux in the image
             x = circumstellar flux
     """

    def __init__(self, rot_angles: np.array, coro: np.array, psf: None or np.array):

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

    def forward_ADI(self, L: torch.Tensor, x: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * L + R(x) )  """

        if flux is None: flux = torch.ones(self.nb_frame - 1)
        if fluxR is None: flux = torch.ones(self.nb_frame - 1)

        Y = torch.zeros((self.nb_frame,) + L.shape).double()

        # First image. No intensity vector
        Rx = tensor_rotate_fft(ReLU(x), float(self.rot_angles[0]))
        Y[0] = ReLU(L) + Rx

        for frame_id in range(1, self.nb_frame):
            Rx = tensor_rotate_fft(ReLU(x), float(self.rot_angles[frame_id]))
            if self.psf is not None: Rx = tensor_conv(Rx, self.psf)
            Y[frame_id] = flux[frame_id - 1] * ReLU(L) + fluxR[frame_id - 1] * Rx

        return Y

    def forward_ADI_reverse(self, L: torch.Tensor, x: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * R(L) + x) )  """

        if flux is None: flux = torch.ones(self.nb_frame - 1)
        if fluxR is None: flux = torch.ones(self.nb_frame - 1)

        Y = torch.zeros((self.nb_frame,) + L.shape).double()

        # First image. No intensity vector
        Rl = tensor_rotate_fft(ReLU(L), -float(self.rot_angles[0]))
        Y[0] = Rl + self.coro * ReLU(x)

        for frame_id in range(1, self.nb_frame):
            RL = tensor_rotate_fft(ReLU(L), -float(self.rot_angles[frame_id]))
            if self.psf is not None: x = tensor_conv(ReLU(x), self.psf)
            Y[frame_id] = flux[frame_id - 1] * RL + fluxR[frame_id - 1] * ReLU(x)

        return Y

    def get_Rx(self, x: torch.Tensor, flux=None) -> torch.Tensor:

        Rx = torch.zeros((self.nb_frame,) + self.frame_shape).double()
        if flux is None: flux = torch.ones(self.nb_frame - 1)
        Rx[0] = self.coro * tensor_rotate_fft(ReLU(x), float(self.rot_angles[0]))

        for frame_id in range(1, self.nb_frame):
            Rx[frame_id] = flux[frame_id - 1] * self.coro * tensor_rotate_fft(ReLU(x), float(self.rot_angles[frame_id]))

        return Rx


def pad_psf(M: numpy.array, shape) -> numpy.array:
    """Pad with 0 the PSF if it is too small """
    if M.shape[0] % 2:
        raise ValueError("PSF shape should be wise : psf must be centred on 4pix not one")
    M_resized = np.zeros(shape)
    mid = shape[0] // 2
    psfmid = M.shape[0] // 2
    M_resized[mid - psfmid:mid + psfmid, mid - psfmid:mid + psfmid] = M

    return M_resized
