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

# FFT tensor operator
from mustard.algo import tensor_conv, tensor_rotate_fft, tensor_fft_scale

# Regular interpolated tensor operators
from torchvision.transforms.functional import rotate
from torchvision.transforms.functional import scale as scaletf
from torchvision.transforms.functional import InterpolationMode

# ReLu
from torch.nn import ReLU as reluConstr
ReLU = reluConstr()

# I used this to switch rotation methods
# Could have been added as an option or chose the one to keep...
tensor_rotate =  lambda frame, angle : rotate(frame, angle, InterpolationMode.BILINEAR) # OR tensor_rotate_fft(frame, angle)
tensor_scale = lambda frame, scale : scaletf(frame, scale) # OR tensor_fft_scale(frame, scale)

# %% Mother class

class Cube_model():

    def __init__(self, nb_frame: int, coro: np.array, psf: None or np.array):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- Constants
        # Sizes
        self.frame_shape = coro.shape
        self.nb_frame = nb_frame

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
        self.rot_angles = rot_angles
        self.nb_rframe = len(rot_angles)

        super().__init__(self.nb_rframe, coro, psf)

    def forward(self, L: torch.Tensor, x: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * L + R(x) )  """

        if flux is None: flux = torch.ones(self.nb_rframe - 1).double().to(self.device)
        if fluxR is None: fluxR = torch.ones(self.nb_rframe - 1).double().to(self.device)

        Y = torch.zeros((self.nb_rframe,) + L.shape).double().to(self.device)

        # First image. No intensity vector
        Rx = tensor_rotate(ReLU(x), float(self.rot_angles[0]))
        Y[0] =  ReLU(L + Rx)

        for frame_id in range(1, self.nb_rframe):
            Rx = tensor_rotate(ReLU(x), float(self.rot_angles[frame_id]))
            if self.psf is not None: Rx = tensor_conv(Rx, self.psf)
            Y[frame_id] =  ReLU(flux[frame_id - 1] * (fluxR[frame_id - 1] * ReLU(L) + Rx))

        return Y

    def forward_ADI_reverse(self, L: torch.Tensor, x: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * R(L) + x) )  """

        if flux is None: flux = torch.ones(self.nb_rframe - 1).double().to(self.device)
        if fluxR is None: fluxR = torch.ones(self.nb_rframe - 1).double().to(self.device)

        Y = torch.zeros((self.nb_rframe,) + L.shape).double().to(self.device)

        # First image. No intensity vector
        Rl = tensor_rotate(L, -float(self.rot_angles[0]))
        Y[0] = ReLU(Rl) + self.coro * ReLU(x)

        for frame_id in range(1, self.nb_rframe):
            RL = tensor_rotate(ReLU(L), -float(self.rot_angles[frame_id]))
            if self.psf is not None: x = tensor_conv(ReLU(x), self.psf)
            Y[frame_id] = ReLU(flux[frame_id - 1] * (fluxR[frame_id - 1] * ReLU(RL) + ReLU(x)))

        return Y

    def get_Lf(self, L: torch.Tensor, flux=None, fluxR=None, rot=False) -> torch.Tensor:

        Lf = torch.zeros((self.nb_rframe, 1) + self.frame_shape).double().to(self.device)
        if flux is None: flux = torch.ones(self.nb_rframe - 1).double().to(self.device)
        if fluxR is None: fluxR = torch.ones(self.nb_rframe - 1).double().to(self.device)
        Lf[0] = ReLU(L)

        if rot:
            Lf[0] = tensor_rotate(ReLU(L), float(self.rot_angles[0]))
            for frame_id in range(1, self.nb_rframe):
                Lf[frame_id] = fluxR[frame_id - 1] * flux[frame_id - 1] * \
                               tensor_rotate(ReLU(L), float(self.rot_angles[frame_id]))
            return Lf

        else :
            for frame_id in range(1, self.nb_rframe):
                Lf[frame_id] = fluxR[frame_id - 1] * flux[frame_id - 1] * ReLU(L)

            return  Lf

    def get_Rx(self, x: torch.Tensor, flux=None, inverse=False) -> torch.Tensor:

        sgn = -1 if inverse else 1
        Rx = torch.zeros((self.nb_rframe, 1) + self.frame_shape).double().to(self.device)
        if flux is None: flux = torch.ones(self.nb_rframe - 1).double().to(self.device)
        Rx[0] = tensor_rotate(ReLU(x), sgn*float(self.rot_angles[0]))

        for frame_id in range(1, self.nb_rframe):
            Rx[frame_id] =  ReLU(flux[frame_id - 1] * tensor_rotate(ReLU(x), sgn*float(self.rot_angles[frame_id])))

        return Rx


# %% Forward ASDI model :
    
class model_ASDI(Cube_model):
    """ Forward models as presented in mayo

            Y = scale_k(L) + R_j(x)

        with L = starlight/constant flux in the image
             x = circumstellar flux
             k, j id of spectral/angular diversity
     """

    def __init__(self, rot_angles: np.array, scales: np.array, coro: np.array, psf: None or np.array):

        self.scales = scales
        self.rot_angles = rot_angles
        self.nb_rframe = len(rot_angles)
        self.nb_sframe = len(scales)

        super().__init__(self.nb_rframe, coro, psf)


    def forward(self, L: torch.Tensor, x: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * L + R(x) )  """

        # TODO flux can also vary between spectraly diverse frames ??
        if flux is None: flux = torch.ones(self.nb_rframe - 1).double().to(self.device)
        if fluxR is None: fluxR = torch.ones(self.nb_rframe - 1).double().to(self.device)

        Y = torch.zeros((self.nb_rframe, self.nb_sframe) + L.shape).double().to(self.device)

        # First image. No intensity vector
        Rx = tensor_rotate(ReLU(x), float(self.rot_angles[0]))
        Sl = tensor_scale(ReLU(L), float(self.scales[0]))
        Y[0] = Sl + Rx

        for id_r in range(1, self.nb_rframe):
            for id_s in range(0, self.nb_sframe):
                Rx = tensor_rotate(ReLU(x), float(self.rot_angles[id_r]))
                Sl = tensor_scale(ReLU(L), float(self.scales[id_s]))
                if self.psf is not None: Rx = tensor_conv(Rx, self.psf)
                Y[id_r, id_s] = flux[id_r - 1] * ( fluxR[id_r - 1] * Sl + Rx )

        return Y


# %% Forward SDI model :
    
class model_SDI(Cube_model):
    """ Forward models as presented in mayo

            Y = scale_k(L) + X

        with L = starlight/constant flux in the image
             x = circumstellar flux
             k, j id of spectral/angular diversity
     """

    def __init__(self, scales: np.array, coro: np.array, psf: None or np.array):

        self.scales = scales
        self.nb_sframe = len(scales)
        super().__init__(self.nb_sframe, coro, psf)



    def forward(self, L: torch.Tensor, x: torch.Tensor, flux=None, fluxR=None) -> torch.Tensor:
        """ Process forward model  : Y =  ( flux * L + R(x) )  """

        if flux is None: flux = torch.ones(self.nb_sframe - 1).double().to(self.device)
        if fluxR is None: fluxR = torch.ones(self.nb_sframe - 1).double().to(self.device)

        Y = torch.zeros((self.nb_sframe, ) + L.shape).double().to(self.device)

        # First image. No intensity vector
        Sl = tensor_scale(ReLU(L), 1/float(self.scales[0]))
        Y[0] = Sl + ReLU(x)

        for id_s in range(1, self.nb_sframe):
            Sl = tensor_scale(ReLU(L), 1/float(self.scales[id_s]))
            Y[id_s] = flux[id_s - 1] * ( fluxR[id_s - 1] * Sl + ReLU(x) )

        return Y
    

# %% Gauss 2D Model (stellar halo model) :
    
class Gauss_2D():
    """ Torch implementation of Gaussian modeling.
    Inspired from astropy.modeling.Fittable2DModel Gaussian2D """

    def __init__(self, img: np.ndarray, mask=1, exceeding=0.2):

        self.cent = img.shape[0]//2
        self.indices   = torch.from_numpy(np.indices(img.shape))
        x, y = (self.indices).numpy()

        from scipy.optimize import (fmin_slsqp, minimize) # SLSQP, not very robust, minimiz truct better
        from scipy.optimize import NonlinearConstraint

        from torchmin import minimize_constr # troch wrapper for trust_const but it is beta :(
        from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
        from vip_hci.var import fit_2dgaussian, fit_2d2gaussian

        # Initilisation
        amplitude = np.max(mask*img) - np.min(img)
        x_stddev = 4*gaussian_fwhm_to_sigma
        y_stddev = 4*gaussian_fwhm_to_sigma
        theta = 1

        param = [amplitude, x_stddev, y_stddev, theta]
        imgT = torch.from_numpy(img)
        cons_max = img
        cons_min = 0

        def model(param):
            amplitude, x_stddev, y_stddev, theta = param

            cost2 = np.cos(theta) ** 2
            sint2 = np.sin(theta) ** 2
            sin2t = np.sin(2. * theta)
            xstd2 = x_stddev ** 2
            ystd2 = y_stddev ** 2
            xdiff = x - self.cent
            ydiff = y - self.cent
            a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
            b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
            c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))

            return np.abs(amplitude) * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                                              (c * ydiff ** 2)))

        def constrain_max(param):
            """Condition : cons_max - model(param) >= 0 -> cons_max >= model(param)"""
            return mask*((1+exceeding)*cons_max - model(param))

        def constrain_min(param):
            """Condition: model(param) - cons_min >= 0 -> cons_min <= model(param)"""
            return mask*(model(param) - cons_min)

        def f_const(param):
            """Wrap the constrains functions"""
            return np.ravel([constrain_max(param)])

        def weight_model(param):
            return mask*model(param)

        def objective_function(param):
            """Distance to the model"""
            return np.linalg.norm(mask*(abs(img) - model(param)), 2)**2

        res = fit_2dgaussian(img, full_output=True, debug=False)
        amplitude, theta  = res['amplitude'][0], res['theta'][0]
        x_stddev, y_stddev = res['fwhm_x'][0] * gaussian_fwhm_to_sigma, res['fwhm_y'][0] * gaussian_fwhm_to_sigma

        self.amplitude = torch.FloatTensor([amplitude])
        self.x_stddev  = torch.FloatTensor([x_stddev])
        self.y_stddev  = torch.FloatTensor([y_stddev])
        self.theta     = np.deg2rad(torch.FloatTensor([theta]))
        self.Hinit_no_mask = self.generate()

        # res = fit_2d2gaussian(img*mask, full_output=True, debug=False)
        res = fit_2dgaussian(mask*abs(img), full_output=True, debug=False, weights=mask)
        amplitude, theta  = res['amplitude'][0], res['theta'][0]
        x_stddev, y_stddev = res['fwhm_x'][0] * gaussian_fwhm_to_sigma, res['fwhm_y'][0] * gaussian_fwhm_to_sigma

        self.amplitude = torch.FloatTensor([amplitude])
        self.x_stddev  = torch.FloatTensor([x_stddev])
        self.y_stddev  = torch.FloatTensor([y_stddev])
        self.theta     = np.deg2rad(torch.FloatTensor([theta]))

        self.Hinit = self.generate()

        print("Estimation of Halo .. \n")
        amplitude_tresh = amplitude/2
        param = np.array([amplitude_tresh, x_stddev, y_stddev, theta])
        # cons_dict = {'fun':weight_model, 'lb':weights*cons_min, 'ub':weights*cons_max}
        # cons_obj = NonlinearConstraint(weight_model, weights*cons_min, weights*cons_max)
        res = fmin_slsqp(objective_function, param, f_ieqcons=f_const)
        # res = minimize(objective_function, param, method='trust-constr', constraints=cons_obj)
        # res = minimize_constr(objective_function, param,constr=cons_dict)
        # print("Done")

        amplitude, x_stddev, y_stddev, theta = res
        self.amplitude = torch.FloatTensor([amplitude])
        self.x_stddev  = torch.FloatTensor([x_stddev])
        self.y_stddev  = torch.FloatTensor([y_stddev])
        self.theta     = np.deg2rad(torch.FloatTensor([theta]))

    def generate_k(self, amplitude=None, x_stddev=None, y_stddev=None, theta=None):

        x, y = self.indices
        amplitude_k = amplitude if amplitude is not None else self.amplitude
        x_stddev_k  = x_stddev  if x_stddev  is not None else self.x_stddev
        y_stddev_k  = y_stddev  if y_stddev  is not None else self.y_stddev
        theta_k     = theta     if theta     is not None else self.theta

        cost2 = torch.cos(theta_k) ** 2
        sint2 = torch.sin(theta_k) ** 2
        sin2t = torch.sin(2. * theta_k)
        xstd2 = x_stddev_k ** 2
        ystd2 = y_stddev_k ** 2
        xdiff = x - self.cent
        ydiff = y - self.cent
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))

        return torch.abs(amplitude_k) * torch.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                                    (c * ydiff ** 2)))

    def generate(self):
        x, y = self.indices

        cost2 = torch.cos(self.theta) ** 2
        sint2 = torch.sin(self.theta) ** 2
        sin2t = torch.sin(2. * self.theta)
        xstd2 = self.x_stddev ** 2
        ystd2 = self.y_stddev ** 2
        xdiff = x - self.cent
        ydiff = y - self.cent
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))

        return np.abs(self.amplitude) * torch.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                                            (c * ydiff ** 2)))


def pad_psf(M: np.array, shape) -> np.array:
    """Pad with 0 the PSF if it is too small """
    if M.shape[0] % 2:
        raise ValueError("PSF shape should be wise : psf must be centred on 4pix not one")
    M_resized = np.zeros(shape)
    mid = shape[0] // 2
    psfmid = M.shape[0] // 2
    M_resized[mid - psfmid:mid + psfmid, mid - psfmid:mid + psfmid] = M

    return M_resized
