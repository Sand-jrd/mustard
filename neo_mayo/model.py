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

    def __init__(self, rot_angles: np.array, mask: np.array, rot="fft"):

        # -- Constants

        # Frames known rotations list
        self.rot_angles = rot_angles

        # Pupil mask
        self.mask = torch.unsqueeze(torch.from_numpy(mask), 0)

        # Sizes
        self.nb_frame = len(rot_angles)
        self.frame_shape = mask.shape

        # -- Functions

        if rot == "fft":
            self.rot = tensor_rotate_fft
            self.rot_args = {}

        else:
            self.rot = rotate  # Torchvison rotation
            self.rot_args = {'interpolation': InterpolationMode.BILINEAR}

        self.conv = lambda x, y, **kwargs: torch.abs(torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.fftshift(torch.fft.fft2(x)) * torch.fft.fftshift(torch.fft.fft2(y)))))

    def forward_ADI(self, L, x, flux=None):
        """ Process forward model  : Y =  ( l_i * L + R(x)) )  """

        if flux is None: flux = torch.ones(self.nb_frame - 1)

        Y = torch.zeros((self.nb_frame,) + L.shape).double()

        # First image. No intensity vector
        Rx = self.rot(x.abs(), float(self.rot_angles[0]), **self.rot_args)
        Y[0] = L + Rx

        for frame_id in range(1, self.nb_frame):
            Rx = self.rot(x.abs(), float(self.rot_angles[frame_id]), **self.rot_args)
            Y[frame_id] = flux[frame_id-1]*L + Rx

        return Y
