#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:42:10 2021

______________________________

| Forward model used in mayo |
______________________________

@author: sand-jrd
"""

from vip_hci.preproc import frame_rotate
from scipy.signal import fftconvolve
from scipy.ndimage import rotate as sci_rotate
from astropy.convolution import convolve_fft
import numpy as np
import torch
from torchvision.transforms.functional import rotate, InterpolationMode
from torch.nn.functional import relu
from neo_mayo.utils import sobel_tensor_conv, var_inmatrix, var_inline
from scipy.special import huber

import warnings

warnings.simplefilter("ignore", UserWarning)


# %% Forward ADI model :

class model_ADI:
    """ Forward models as presented in mayo
        
            Y = L + R(x)
        
        with L = starlight/constant flux in the image
             x = circumstellar flux
     """

    def __init__(self, rot_angles, phi_coro, mask, conv="fft", rot_opt="no-vip"):

        self.rot_angles = rot_angles  # Frames known rotations list
        self.phi_coro = phi_coro  # Response of the coronograph
        self.mask = mask  # Pupil mask
        self.torch = False

        self.nb_frame = len(rot_angles)
        self.frame_shape = phi_coro.shape

        # -- Options

        if rot_opt == "fastest":
            self.rotate = frame_rotate
            self.rot_args = {"imlib": 'opencv', "interpolation": 'nearneig'}
        if rot_opt == "best":
            self.rotate = frame_rotate
            self.rot_args = {}
        if rot_opt == "no-vip":
            self.rotate = sci_rotate
            self.rot_args = {"reshape": False}

        if conv == "astro":
            self.conv = convolve_fft
            self.conv_arg = {"normalize_kernel": False, "nan_treatment": 'fill'}
        if conv == "scipy":
            self.conv = fftconvolve
            self.conv_arg = {"mode": "same"}
        if conv == "fft":
            self.conv = lambda x, y, **kwargs: abs(
                np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(x)) * np.fft.fftshift(np.fft.fft2(y)))))
            self.conv_arg = {}

    def adapt_torch(self, delta=1):

        # -- Convert to tensor + unsqueeze adding dummy dimention
        # to fit tensor rotate requirement
        if not self.torch:
            self.phi_coro = torch.unsqueeze(torch.from_numpy(self.phi_coro), 0).float()
            self.mask = torch.unsqueeze(torch.from_numpy(self.mask), 0).float()

            self.hub_loss = torch.nn.HuberLoss('sum', float(delta))
            self.conv = lambda x, y, **kwargs: torch.abs(torch.fft.ifftshift(
                torch.fft.ifft2(torch.fft.fftshift(torch.fft.fft2(x)) * torch.fft.fftshift(torch.fft.fft2(y)))))

        self.torch = True

    def forward_ADI(self, L, x):
        """ Process forward model as discribe in mayo : Y = M * ( L + conv(phi,R(x)) )  """

        Y = np.ndarray((self.nb_frame,) + L.shape)

        for frame_id in range(self.nb_frame):
            Rx = self.rotate(x, self.rot_angles[frame_id], **self.rot_args)
            Y[frame_id] = self.mask * (abs(L) + self.conv(self.phi_coro, Rx, **self.conv_arg))

        return Y

    def forward_torch_ADI(self, L, x):
        """ Process forward model as discribe in mayo : Y = M * ( L + conv(phi,R(x)) )  """

        Y = torch.zeros((self.nb_frame,) + L.shape)

        for frame_id in range(self.nb_frame):
            Rx = torch.abs(rotate(x, float(self.rot_angles[frame_id]), interpolation=InterpolationMode.BILINEAR))
            # conv_Rx = self.conv(self.phi_coro,x)
            Y[frame_id] = self.mask * (torch.abs(L) + Rx.abs())

        return Y


# %% Constrain definitions 

def create_bounds(constantes):
    science_data = constantes["science_data"]
    vmin = np.min(science_data, axis=None)
    vmax = np.max(science_data, axis=None)

    size = science_data[0].size

    return ((vmin, vmax),) * 2 * size


def create_constrain_list(model, constantes):
    """ Create the list of model constraints """

    cons = [{'type': 'eq', 'fun': proj_L, 'args': (model, constantes)}]

    return cons


def proj_L(var, model, constantes):
    """ L contrain : Projection of L have to be equal to L"""

    L, x = var_inmatrix(var, model.frame_shape[0])

    proj = constantes["L_proj"]

    return np.sum((proj @ L) - L)


# %% Loss functions and wrappers


def call_loss_function(var, model, constantes):
    """
       Unwrap minimiz parameters and call adi model loss
       
       Parameters
       ----------
       var : array
           inline varaibles (matching the minimiz syntaxs)
       
       model : neo-mayo.model.model_ADI
           Forward model

       constantes : dict
        dictornay containing constante to the minimisation problem

       Returns
       -------
       loss : float
           Value of loss function (see neo_mayo.model.adi_model_loss)
       
    """
    # Unwrap varaible
    L, x = var_inmatrix(var, model.frame_shape[0])

    loss = adi_model_loss(model, L, x, constantes)

    if constantes["regul"]: loss += regul_L(L, constantes)

    return loss


def adi_model_loss(model, L, x, constantes):
    """ ADI loss models as discribe in mayo
     Loss models of the model 
     
     loss = huberLoss( Y - (M * (L + conv(phi,R(x)) ) )
     
    Parameters
    ----------

    model : neo-mayo.model.model_ADI
        Forward model (M * (L + conv(phi,R(x)) ) )

    L,x : ndarray
     inputs

    constantes : dict
        dictornay containing constante to the minimisation problem
            * science-data -> science data inputs
            * delta -> indicating the quadratic vs. linear loss changepoint of huber loss
       
       Returns
       -------
       loss : float
           Value of loss function (see neo_mayo.model.adi_model_loss)
       
    """

    # Unpack constantes
    science_data = constantes["science_data"]
    delta = constantes["delta"]

    #  Compute forward model
    Y = model.forward_ADI(L, x)

    # Compute model loss with huber distance
    loss = huber(delta, Y - science_data)

    return np.sum(loss)


def adi_model_loss_torch(model, L, x, constantes):
    """  same as normal but tensor freidnly  """

    # Unpack constantes
    science_data = torch.unsqueeze(torch.from_numpy(constantes["science_data"]), 1).float()

    #  Compute forward model
    Y = model.forward_torch_ADI(L, x)

    # Compute model loss with huber distance
    loss = model.hub_loss(Y, science_data)
    R = regul_L_torch(L, constantes)

    return loss + R


def call_loss_grad(var, model, constantes):
    """
       Unwrap minimiz parameters and call adi model loss
       
       Parameters
       ----------
       var : array
           inline varaibles (matching the minimiz syntaxs)
       
       model : neo-mayo.model.model_ADI
           Forward model

       constantes : dict
        dictornay containing constante to the minimisation problem

       Returns
       -------
       loss : float
           Value of loss function (see neo_mayo.model.adi_model_loss)
       
    """
    # Unwrap varaible
    L, x = var_inmatrix(var, model.frame_shape[0])

    return var_inline(loss_grad(model, L, x, constantes))


def loss_grad(model, L, x, constantes):
    """ Process backward model xich is a simple order 1 différential
    approximation gradient
    
    """
    # Store gradient 
    gradL = np.zeros(L.shape)
    gradX = np.zeros(x.shape)

    # delta to derive
    dL = np.zeros(L.shape)
    dX = np.zeros(x.shape)

    # We try each strings d_x, d_y to compute gradients
    for d_x in range(L.shape[0]):
        for d_y in range(L.shape[0]):
            # Grad de L
            dL[d_x, d_y] = 1
            gradL += adi_model_loss(model, dL, dX, constantes)
            dL[d_x, d_y] = 0

            # Grad de X
            dX[d_x, d_y] = 1
            gradX += adi_model_loss(model, dL, dX, constantes)
            dX[d_x, d_y] = 0

    return gradL, gradX


def regul_L(L, constantes):
    """ Loss function on L prior """

    proj = constantes["L_proj"]
    mu = constantes["hyper_p"]

    A = proj @ L

    return mu * np.sum((A - L) ** 2)


def regul_L_torch(L, constantes):
    """ Loss function on L prior """

    proj = constantes["L_proj"]
    mu = constantes["hyper_p"]

    A = proj @ L

    return float(mu) * torch.sum((A[0] - L[0]) ** 2)


def regul_X(x):
    """ Loss function on x prior """
    return torch.norm(relu(sobel_tensor_conv(x), inplace=True))
