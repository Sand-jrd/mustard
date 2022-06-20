#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:42:10 2021

______________________________

| Forward model used in mayo |
______________________________

@author: sand-jrd
"""

import numpy as np
import torch
from mustard.algo import tensor_rotate_fft, tensor_conv, tensor_fft_scale
from torch.nn import ReLU as reluConstr
ReLU = reluConstr()

class Cube_model():

    def __init__(self, rot_angles: np.array, coro: np.array, psf: None or np.array):

        # -- Constants
        # Sizes
        self.nb_rframe = len(rot_angles)
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

    def forward(self):
        # If I was a good programmer I would have writen the assertions to prevent bugs here..
        # I'll do it later maybe
        return None

# %% Forward ADI model :
class model_ADI(Cube_model):
    """ Forward models
        
            Y = L + R(x)
        
        with L = starlight/constant flux in the image
             x = circumstellar flux
     """

    def __init__(self, rot_angles: np.array, coro: np.array, psf: None or np.array):
        super().__init__(rot_angles, coro, psf)

    def forward(self, L: torch.Tensor, x: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * L + R(x) )  """

        if flux is None: flux = torch.ones(self.nb_rframe - 1)
        if fluxR is None: fluxR = torch.ones(self.nb_rframe - 1)

        Y = torch.zeros((self.nb_rframe,) + L.shape).double()

        # First image. No intensity vector
        Rx = tensor_rotate_fft(ReLU(x), float(self.rot_angles[0]))
        Y[0] =  ReLU(L + Rx)

        for frame_id in range(1, self.nb_rframe):
            Rx = tensor_rotate_fft(ReLU(x), float(self.rot_angles[frame_id]))
            if self.psf is not None: Rx = tensor_conv(Rx, self.psf)
            Y[frame_id] =  ReLU(flux[frame_id - 1] * (fluxR[frame_id - 1] * ReLU(L) + Rx))

        return Y

    def forward_ADI_reverse(self, L: torch.Tensor, x: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * R(L) + x) )  """

        if flux is None: flux = torch.ones(self.nb_rframe - 1)
        if fluxR is None: fluxR = torch.ones(self.nb_rframe - 1)

        Y = torch.zeros((self.nb_rframe,) + L.shape).double()

        # First image. No intensity vector
        Rl = tensor_rotate_fft(L, -float(self.rot_angles[0]))
        Y[0] = ReLU(Rl) + self.coro * ReLU(x)

        for frame_id in range(1, self.nb_rframe):
            RL = tensor_rotate_fft(ReLU(L), -float(self.rot_angles[frame_id]))
            if self.psf is not None: x = tensor_conv(ReLU(x), self.psf)
            Y[frame_id] = ReLU(flux[frame_id - 1] * (fluxR[frame_id - 1] * ReLU(RL) + ReLU(x)))

        return Y

    def get_Lf(self, L: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:

        Lf = torch.zeros((self.nb_rframe, 1) + self.frame_shape).double()
        if flux is None: flux = torch.ones(self.nb_rframe - 1)
        if fluxR is None: fluxR = torch.ones(self.nb_rframe - 1)
        Lf[0] = ReLU(L)

        for frame_id in range(1, self.nb_rframe):
            Lf[frame_id] = fluxR[frame_id - 1] * flux[frame_id - 1] * ReLU(L)

        return  Lf

    def get_Rx(self, x: torch.Tensor, flux=None, inverse=False) -> torch.Tensor:

        sgn = -1 if inverse else 1
        Rx = torch.zeros((self.nb_rframe, 1) + self.frame_shape).double()
        if flux is None: flux = torch.ones(self.nb_rframe - 1)
        Rx[0] = tensor_rotate_fft(ReLU(x), sgn*float(self.rot_angles[0]))

        for frame_id in range(1, self.nb_rframe):
            Rx[frame_id] =  ReLU(flux[frame_id - 1] * tensor_rotate_fft(ReLU(x), sgn*float(self.rot_angles[frame_id])))

        return Rx


# %% Forward ADI model :
class model_ASDI(Cube_model):
    """ Forward models as presented in mayo

            Y = scale_k(L) + R_j(x)

        with L = starlight/constant flux in the image
             x = circumstellar flux
             k, j id of spectral/angular diversity
     """

    def __init__(self, rot_angles: np.array, scales: np.array, coro: np.array, psf: None or np.array):
        super().__init__(rot_angles, coro, psf)
        self.scales = scales
        self.nb_sframe = len(scales)


    def forward(self, L: torch.Tensor, x: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * L + R(x) )  """

        # TODO flux can also vary between spectraly diverse frames ??
        if flux is None: flux = torch.ones(self.nb_rframe - 1)
        if fluxR is None: fluxR = torch.ones(self.nb_rframe - 1)

        Y = torch.zeros((self.nb_rframe, self.nb_sframe) + L.shape).double()

        # First image. No intensity vector
        Rx = tensor_rotate_fft(ReLU(x), float(self.rot_angles[0]))
        Sl = tensor_fft_scale(ReLU(L), float(self.scales[0]))
        Y[0] = Sl + Rx

        for id_r in range(1, self.nb_rframe):
            for id_s in range(0, self.nb_sframe):
                Rx = tensor_rotate_fft(ReLU(x), float(self.rot_angles[id_r]))
                Sl = tensor_fft_scale(ReLU(L), float(self.scales[id_s]))
                if self.psf is not None: Rx = tensor_conv(Rx, self.psf)
                Y[id_r,id_s] = flux[id_r - 1] * ( fluxR[id_r - 1] * Sl + Rx )

        return Y


def pad_psf(M: np.array, shape) -> np.array:
    """Pad with 0 the PSF if it is too small """
    if M.shape[0] % 2:
        raise ValueError("PSF shape should be wise : psf must be centred on 4pix not one")
    M_resized = np.zeros(shape)
    mid = shape[0] // 2
    psfmid = M.shape[0] // 2
    M_resized[mid - psfmid:mid + psfmid, mid - psfmid:mid + psfmid] = M

    return M_resized
