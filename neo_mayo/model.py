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

    def __init__(self, rot_angles: np.array, phi_coro: np.array, mask: np.array, rot="fft"):

        # -- Constants

        # Frames known rotations list
        self.rot_angles = rot_angles

        # Response of the coronograph
        self.phi_coro = torch.unsqueeze(torch.from_numpy(phi_coro), 0)

        # Pupil mask
        self.mask = torch.unsqueeze(torch.from_numpy(mask), 0)

        # Sizes
        self.nb_frame = len(rot_angles)
        self.frame_shape = phi_coro.shape

        # -- Functions

        if rot == "fft":
            self.rot = tensor_rotate_fft
            self.rot_args = {}

        else:
            self.rot = rotate  # Torchvison rotation
            self.rot_args = {'interpolation': InterpolationMode.BILINEAR}

        self.conv = lambda x, y, **kwargs: torch.abs(torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.fftshift(torch.fft.fft2(x)) * torch.fft.fftshift(torch.fft.fft2(y)))))

    def forward_ADI(self, L, x):
        """ Process forward model as describe in mayo : Y = M * ( L + conv(phi,R(x)) )  """

        Y = torch.zeros((self.nb_frame,) + L.shape).double()

        for frame_id in range(self.nb_frame):
            Rx = torch.abs(self.rot(x, float(self.rot_angles[frame_id]), **self.rot_args))
            # conv_Rx = self.conv(self.phi_coro,Rx)
            Y[frame_id] = torch.abs(L) + Rx.abs()

        return Y
