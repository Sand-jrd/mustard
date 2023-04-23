#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:30:10 2021

______________________________

|      Neo-Mayo estimator     |
______________________________


@author: sand-jrd
"""

# -- For file management -- #
from vip_hci.fits import write_fits
from os import makedirs, remove, rmdir
from os.path import isdir

# -- Algo and science model -- #
from mustard.model import model_ADI, model_ASDI, model_SDI
from mustard.algo import sobel_tensor_conv, convert_to_mask, radial_profil, res_non_convexe, create_radial_prof_matirx

# Numpy operators                          
from vip_hci.preproc import cube_derotate, frame_rotate,cube_rescaling_wavelengths, cube_crop_frames
# Torch operators
from torchvision.transforms.functional import rotate
from torchvision.transforms.functional import InterpolationMode

# -- Loss functions -- #
import torch.optim as optim
from torch import sum as tsum
from torch.nn import ReLU as relu_constr

# -- Misk -- #
import torch
import numpy as np
from mustard.utils import circle, iter_to_gif, print_iter
from copy import deepcopy

# -- For verbose -- #
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
from PIL import Image



# %% Verbose material

sep = ('_' * 50)
head = "|it |       loss        |        R1        |        R2        |       total      |"
info_iter = "|{:^3}|{:.6e} ({:^3.0f}%)|{:.6e} ({:^2.0f}%)|{:.6e} ({:^2.0f}%)|{:.12e}|"

init_msg = sep + "\nResolving IP-ADI optimization problem - name : {}" + \
           "\n Outputs will be saved {}" + \
           "\nRegul R1 : '{}' and R2 : '{}'" + \
           "\n{} deconvolution and {} frame weighted based on rotations" + \
           "\nRelative amplitude of {} will be estimated" + \
           "\nRegul weight are set to w_r={:.2f}% and w_r2={:.2f}%, maxiter={}\n"

activ_msg = "REGUL HAVE BEEN {} with w_r={:.2f} and w_r2={:.2f}"

ReLU = relu_constr()

def loss_ratio(Ractiv: int or bool, R1: float, R2: float, L: float) -> tuple:
    """ Compute Regul weight over data attachment terme """
    return tuple(np.array((1, 100 / L)) * (L - Ractiv * (abs(R1) + abs(R2)))) + \
           tuple(np.array((1, 100 / L)) * abs(R1)) + \
           tuple(np.array((1, 100 / L)) * abs(R2))


# %% ------------------------------------------------------------------------

class mustard_estimator:
    """ MUSTARD Algorithm main class  """

    def __init__(self, science_data: np.ndarray, angles: np.ndarray, scale=None, coro=6, pupil="edge",
                 psf=None, hid_mask=None, Badframes=None, savedir='./', ref=None):
        """
        Initialisation of estimator object

        Parameters
        ----------
        science_data : np.ndarray
            ADI cube. The first dimensions should be the number of frames.

        angles : np.array
            List of angles. Should be the same size as the ADI cube 1st dimension (number of frames)

        scale : np.array or None
            List of scaling coeff for SDI

        coro : int or str
            Size of the coronograph

        pupil : int or None or "edge"
            Size of the pupil.
            If pupil is set to "edge" : the pupil raduis will be half the size of the frame
            If pupil is set to None : there will be no pupil at all

        psf : np.ndarray or None
            If a psf is provided, will perform deconvolution (i.e conv by psf inculded in forward model)
            
        Hid_mask: np.array or None
            Mask to indicate the corrected region.
            Used for coronograph with specific corrected area (i.e : vapp)
            
        Badframes : tuple or list or None
            Bad frames that you will not be taken into account

        savedir : str
            Path for outputs
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # -- Create model and define constants --

        if Badframes is not None:
            science_data = np.delete(science_data, Badframes, 0)
            angles = np.delete(angles, Badframes, 0)
            # scale = np.delete(scale, Badframes, 0)

        # Constants
        self.L0x0 = None
        self.mask = 1  # R2 mask default. Will be set if call R2 config
        self.name = ''
        self.savedir = savedir

        # Add refs to cube if refs are provided.
        if ref is not None :
            self.nb_ref = ref.shape[0]
            science_data = np.concatenate((science_data, ref), axis=0)
        else : self.nb_ref = 0
        
        # Normalize angles list if provided
        rot_angles = None
        if angles is not None:
            rot_angles = normlizangle(angles)
            order = rot_angles.argsort()
            rot_angles = rot_angles[order]
            science_data = science_data[order]
            ## These two commented lines shift angle list in order to minimize the amplitude of rotation during processing
            # self.ang_shift = find_nearest(rot_angles, np.median(rot_angles))
            # rot_angles -= self.ang_shift
        
        # ADI / SDI / ASDI modes selcted based on provided angles & scales
        if scale is not None and angles is not None:
            # Mode ASDI
            self.shape = science_data[0, 0].shape
            self.nb_frame = science_data.shape[0]
            if science_data.shape[0] != len(angles) or science_data.shape[1] != len(scale):
                raise "Length of scales does not match the size of the science-data cube !"
            self.model = model_ASDI(rot_angles, scale, self.coro, psf)
        
        elif angles is None and scale is None:
            # User forgot something..
            raise "List of angles or/and scales are required !"
        
        elif scale is not None:
            # Mode SDI
           self.shape = science_data[0].shape
           self.nb_frame = science_data.shape[0]  
           if self.nb_frame != len(scale) : raise "Length of scales does not match the size of the science-data cube !"
           self.model = model_SDI(scale, self.coro, psf)

        elif angles is not None:
            # Mode ADI
            self.shape = science_data[0].shape
            self.nb_frame = science_data.shape[0]   
            if self.nb_frame != len(angles) : raise "Length of angles does not match the size of the science-data cube !"
            self.model = model_ADI(rot_angles, self.coro, psf)


        # Coro and pupil masks
        pupilR = pupil
        if pupil is None:
            pupil = np.ones(self.shape)
        elif pupil == "edge":
            pupil = circle(self.shape, self.shape[0] / 2)
            pupilR = circle(self.shape, self.shape[0] / 2 - 2)
        elif isinstance(pupil, (int, float)):
            pupil = circle(self.shape, pupil)
            pupilR = circle(self.shape, pupilR - 2)
        else:
            raise ValueError("Invalid pupil key argument. Possible values : {float/int, None, 'edge'}")

        # Pup_bgk used to compute background intensity value using the outer annulus of 15 pixels
        self.pup_bkg = circle(self.shape, self.shape[0]/2) - circle(self.shape, self.shape[0]/2-15)

        # Coro mask + coro mask for regul (bigger to avoid border effect)
        if coro is None: coro = 3
        self.coro_siz = coro
        self.coro = (1 - circle(self.shape, coro)) * pupil
        self.coroR = (1 - circle(self.shape, coro+2)) * pupilR
        self.final_mask = deepcopy(self.coro)
        
        # For specific corrected region. 
        if hid_mask is not None:
            self.hid_mask = True
            self.coro = hid_mask

        
        # Will be filled with weight if anf_weight option is activated
        self.ang_weight = torch.from_numpy(np.ones(self.nb_frame).reshape((self.nb_frame, 1, 1, 1))).double()

        self.coro = torch.from_numpy(self.coro).double().to(self.device)
        self.coroR = torch.from_numpy(self.coroR).double().to(self.device)

        # -- Configure regularization (can be change later, this is defaults parameters)
        self.config = ["smooth", None, 'No' if psf is None else 'With', 'no', 'Both L and X']
        self.configR1(mode="smooth")
        self.configR2(Msk=None, mode="l1", penaliz="X")
        self.configR3(mode="smooth")

        # Mask used for initalization.
        self.ref_mask = torch.Tensor((1 - circle(self.shape, coro + 2)) * circle(self.shape, self.shape[0] / 2 - 2))
        self.ref_mask_siz = [coro + 2, self.shape[0] / 2 - 2]

        # Itilisization of varaibles.
        self.res = None; # Will store full output results in a dict
        self.last_iter = None; # (X_k, L_k, +option_k)
        self.first_iter = None; # (X_0, L_0, +option_0)
        self.final_estim = None # (X_est, L_est, +option_est)
        
        # (todelete)
        self.ambiguities = None;
        self.speckles = None; 
        self.science_data_ori = None  
        self.pup_bkg_id = torch.tensor(np.array(np.where(self.pup_bkg == 1)), dtype=torch.long)
        self.science_data[np.where(self.science_data == 0)] = np.finfo(float).eps

    # (todelete)
    def compute_bkg(self, X):
        return torch.median(X[self.pup_bkg_id])

    def estimate(self, w_r=0.03, w_r2=0.03, w_r3=0.01, w_pcent=True, estimI="Both", med_sub=False, weighted_rot=True,
                 w_way=(0, 1), maxiter=10, gtol=1e-10, kactiv=0, kdactiv=None, save="./", suffix='', gif=False,
                 verbose=False, history=True, init_maxL=False, mask_L=None):
        """ Resole the minimization of probleme neo-mayo
            The first step with pca aim to find a good initialisation
            The second step process to the minimization

        Parameters
        ----------

        init_maxL: bool
            if True, all ambiguities will be set to L : stellar halo/speakles map. (no recommended)
            Default is False.

        w_r : float
            Weight regularization, hyperparameter to control R1 regularization (smooth regul)

        w_r2 : float
            Weight regularization, hyperparameter to control R2 regularization (mask regul)

        estimI : str (beta)
            If "None" : normal minimization
            if "L" : With estimate a flux variation of the speakles map
            if "Frame" : With estimate a flux variation between each frame
            if "Both" : Will estimate both flux variation between each frame and bewteen frame's speakles map

        med_sub : bool
            If True, will proceed to a median subtraction (recommended)

        weighted_rot : bool (beta)
            if True, each frame will be weighted by the delta of rotation.

        w_way : tuple ints
            If (1,0) : ADI model constructed with the cube and rotate R (direct way)
            If (0,1) : ADI model constructed with the derotated cube and rotate L (reverse way)
            iI (1,1) : will do both

        maxiter : int
            Maximum number of optimization steps

        gtol : float
            Gradient tolerance; Set the break point of minimization.
            The break point is define as abs(J_n - J-(n-1)) < gtol

        kactiv : int
            Activate regularization at a specific iteration.
            Allow to be more accurate on the regul/data-attachment term ratio

        kdactiv : int or {"converg"}
            Deactivate regularization at a specific iteration
            If set to "converg", will be deactivated when break point is reach
            Equivalent to re-run a minimization with no regul and L and X initialized at converged value

        save : str or bool
            If True save at ./mustard_out/
            If path is given, save at the given path + ./mustard_out/

        suffix : string
            String suffix to named the simulation outputs

        gif : bool
            If path is given, save each the minimization step as a gif

        verbose : bool
            Activate or deactivate print
        history : bool
            If verbose == True and history == True, information on current iteration will not be overwritten

         Returns
         -------
         L_est, X_est: ndarray
             Estimated starlight (L) and circunstlellar (X) contributions

        """
        # For verbose, saving, configuration info sortage
        estimI = estimI.capitalize()
        self.name = '_' + suffix if suffix else ''
        self.savedir = (save if isinstance(save, str) else ".") + "/mustard_out" + self.name + "/"
        self.config[4] = estimI + "X and L" if estimI == "Both" else estimI
        overwrite = "\n" if history else "\r"
        ending = "undifined"

        # Keep track on history
        mink = 2  # Min number of iter before convergence
        loss_evo = []
        
        # Regularization activation init setting
        Ractiv = 0 if kactiv else 1  # Ractiv : regulazation is curently activated or not
        w_r = w_r if w_r else 0;
        w_r2 = w_r2 if w_r2 else 0;
        w_r3 = w_r3 if w_r3 else 0
        w_rp = np.array([w_r, w_r2, w_r3]) if w_pcent else np.array([0, 0, 0])  # Percent w, to be computed later

        # Compute frame weights for small angles bias
        if isinstance(self.model, model_ADI) and weighted_rot:
            self.config[3] = "with"
            ang_weight = compute_rot_weight(self.model.rot_angles)
            self.ang_weight = torch.from_numpy(ang_weight.reshape((self.nb_frame, 1, 1, 1))).double().to(self.device)
        elif isinstance(self.model, model_ASDI):
            if weighted_rot:
                self.config[3] = "with"
                ang_weight = compute_rot_weight(self.model.rot_angles)
                self.ang_weight = torch.from_numpy(ang_weight.tiles(
                    (self.model.nb_rframe, self.model.nb_sframe, 1, 1))).double().to(self.device)
            else:
                self.ang_weight = torch.from_numpy(
                    np.ones((self.nb_frame, self.model.nb_sframe, 1, 1, 1))).double().to(self.device)
        elif self.ang_weight is None:
            self.ang_weight = torch.from_numpy(np.ones((self.nb_frame, 1, 1, 1))).double().to(self.device)

        # When pixels by their variances. Can be shut down.
        temporal_var = np.var(self.science_data, axis=0)
        temporal_var *= 1 / np.max(temporal_var)
        self.var_pond = 1  # torch.unsqueeze(torch.from_numpy(1 / temporal_var), 0).double().to(self.device)

        # Combine static masks/weights into one signle weight mask.
        self.weight = self.coro * self.var_pond * self.ang_weight

        # ______________________________________
        # Define constantes and convert arry to tensor

        # Med sub
        if mask_L is not None:
            if len(mask_L) == 2 and isinstance(mask_L[0], int) and isinstance(mask_L[1], int):
                self.ref_mask = torch.Tensor(circle(self.shape, mask_L[1]) - circle(self.shape, mask_L[0]))
                self.ref_mask_siz = mask_L
            else:
                raise "mask_L should be an object that contain two int"

        if isinstance(self.model, model_ASDI):
            science_data = torch.unsqueeze(torch.from_numpy(self.science_data), 2).double().to(self.device)
        else:
            science_data = torch.unsqueeze(torch.from_numpy(self.science_data), 1).double().to(self.device)

        # __________________________________
        # Initialisation with max common

        if self.L0x0 is not None and w_way[1] == 0:
            science_data_derot_np = cube_derotate(self.science_data, self.model.rot_angles)
            science_data_derot = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 1).double().to(self.device)
            if verbose: print("Load initialization")
            L0, X0 = self.L0x0[0], self.L0x0[1]
        else:
            # TODO : sceince_data_derot means derotated (if ADI) or descale (if SDI) or both (if ASDI) consider refactor
            if isinstance(self.model, model_ADI):

                if verbose: print("Mode ADI : Max common init in progress... ")

                if self.nb_ref == 0 :
                    science_data_derot_np = cube_derotate(self.science_data, self.model.rot_angles)
                    science_data_derot = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 2).double()

                    med_L = np.median(self.science_data, axis=0).clip(min=0)
                    R_fr = science_data_derot_np - cube_derotate(np.tile(med_L, (self.nb_frame, 1, 1)),
                                                                 -self.model.rot_angles)
                    res_R = np.mean(R_fr, axis=0).clip(min=0)
                    del R_fr
                    # self.L0x0 = res.clip(min=0), np.mean(R_fr, axis=0).clip(min=0)

                    med_R = np.median(science_data_derot_np, axis=0).clip(min=0)
                    L_fr = self.science_data - cube_derotate(np.tile(med_R, (self.nb_frame, 1, 1)), self.model.rot_angles)
                    res_L = np.mean(L_fr, axis=0).clip(min=0)
                    del L_fr
                    # self.L0x0 = np.mean(L_fr, axis=0).clip(min=0), res.clip(min=0)

                    if init_maxL:
                        L0 = med_L.clip(min=0)*(1 - self.ref_mask.numpy()) + res_L*self.ref_mask.numpy()
                        X0 = med_R*self.ref_mask.numpy() #+ res_R.clip(min=0)*(1 - self.ref_mask.numpy())
                    else:
                        L0 = med_L.clip(min=0)*(1 - self.ref_mask.numpy()) #+ res_L*self.ref_mask.numpy()
                        X0 = res_R.clip(min=0)*(1 - self.ref_mask.numpy()) #+ res_R*self.ref_mask.numpy()

                    L0 = med_L * (1 - self.ref_mask.numpy()) + res_L * self.ref_mask.numpy()
                    X0 = med_R * self.ref_mask.numpy() + res_R * (1 - self.ref_mask.numpy())

                else :

                    science_data_derot_np = cube_derotate(self.science_data[:-self.nb_ref], self.model.rot_angles)
                    ref_mean = np.mean(self.science_data[:self.nb_ref])

                    X0 = np.mean(science_data_derot_np-ref_mean) #+ res_R.clip(min=0)*(1 - self.ref_mask.numpy())
                    empty_sd = self.science_data[:-self.nb_ref] - cube_derotate(np.tile(X0, (self.nb_frame, 1, 1)), self.model.rot_angles)
                    L0 = med_L.clip(min=0)*(1 - self.ref_mask.numpy()) + res_L*self.ref_mask.numpy()

                self.L0x0 = L0, X0
                del res_R, res_L, med_R, med_L

            if isinstance(self.model, model_ASDI):
                if verbose: print("Mode ASDI : Max common init in progress... ")
                science_data_derot_np = np.ndarray(self.science_data.shape)
                for channel in range(self.model.nb_sframe):
                    science_data_derot_np[:, channel] = cube_derotate(self.science_data[:, channel],
                                                                      self.model.rot_angles)
                for angles in range(self.model.nb_rframe):
                    tmp_derot, res, _, _, _, _ = cube_rescaling_wavelengths(science_data_derot_np[angles, :],
                                                                            self.model.scales, full_output=True)
                    if science_data_derot_np.shape[-1] > self.shape[0]:
                        science_data_derot_np[angles, :] = cube_crop_frames(tmp_derot, self.shape[0])
                    else:
                        science_data_derot_np[angles, :] = tmp_derot

                if self.L0x0 is not None:
                    if verbose: print("Load initialization")
                elif init_maxL:
                    res = np.mean(self.science_data, axis=(0, 1))
                    L_fr = np.zeros(science_data_derot_np.shape)
                    res_cube = np.tile(res, (self.model.nb_rframe, self.model.nb_sframe, 1, 1))
                    for channel in range(self.model.nb_sframe):
                        L_fr[:, channel] = cube_derotate(res_cube[:, channel], -self.model.rot_angles)
                    for angles in range(self.model.nb_rframe):
                        tmp_derot, _, _, _, _, _ = cube_rescaling_wavelengths(res_cube[angles, :],
                                                                              self.model.scales, full_output=True)
                        if L_fr.shape[-1] > self.shape[0]: L_fr[angles, :] = cube_crop_frames(tmp_derot, self.shape[0])
                    L_lr = self.science_data - L_fr
                    self.L0x0 = np.mean(L_lr, axis=(0, 1)), res.clip(min=0)
                else:
                    res = np.mean(science_data_derot_np, axis=(0, 1))
                    R_fr = np.zeros(science_data_derot_np.shape)
                    res_cube = np.tile(res, (self.model.nb_rframe, self.model.nb_sframe, 1, 1))
                    for channel in range(self.model.nb_sframe):
                        R_fr[:, channel] = cube_derotate(res_cube[:, channel],
                                                         - self.model.rot_angles)
                    for angles in range(self.model.nb_rframe):
                        tmp_derot, _, _, _, _, _ = cube_rescaling_wavelengths(res_cube[angles, :],
                                                                              1 / self.model.scales, full_output=True)
                        if R_fr.shape[-1] > self.shape[0]: R_fr[angles, :] = cube_crop_frames(tmp_derot, self.shape[0])
                    R_lr = self.science_data - R_fr
                    self.L0x0 = np.mean(R_lr, axis=(0, 1)), res.clip(min=0)
                science_data_derot = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 2).double().to(
                    self.device)

                if isinstance(self.model, model_SDI):
                    if verbose: print("Mode SDI : Max common init in progress... ")
                    science_data_derot_np, res, _, _, _, _ = cube_rescaling_wavelengths(self.science_data,
                                                                                        self.model.scales,
                                                                                        full_output=True, )
                    if science_data_derot_np.shape[-1] > self.shape[0]:
                        science_data_derot_np = cube_crop_frames(science_data_derot_np, self.shape[0])

                    if self.L0x0 is not None:
                        if verbose: print("Load initialization")
                    elif init_maxL:
                        res = np.mean(science_data_derot_np, axis=0)
                        L_fr, L_mean, _, _, _, _ = cube_rescaling_wavelengths(
                            np.tile((res), (self.model.nb_sframe, 1, 1)), 1 / self.model.scales, full_output=True)
                        if L_fr.shape[-1] > self.shape[0]: L_fr = cube_crop_frames(L_fr, self.shape[0])
                        write_fits("L_fr", L_fr)
                        L_lr = (science_data_derot_np - L_fr).clip(min=0)
                        write_fits("L_lr", L_lr)
                        self.L0x0 = np.mean(L_lr, axis=0), res.clip(min=0)
                    else:
                        res = np.median(self.science_data, axis=0)
                        R_fr, R_mean, _, _, _, _ = cube_rescaling_wavelengths(
                            np.tile((res).clip(min=0), (self.model.nb_sframe, 1, 1)), self.model.scales,
                            full_output=True)
                        if R_fr.shape[-1] > self.shape[0]:  R_fr = cube_crop_frames(R_fr, self.shape[0])
                        write_fits("R_fr", R_fr)
                        R_lr = (science_data_derot_np - R_fr).clip(min=0)
                        write_fits("R_lr", R_lr)
                        self.L0x0 = np.mean(R_lr, axis=0), res.clip(min=0)
                    science_data_derot = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 1).double().to(
                        self.device)

        self.get_initialisation(save=True)
        if self.config[0] == "peak_preservation": self.xmax = np.max(X0)

        # __________________________________
        # Initialisation with max common

        L0 = torch.unsqueeze(torch.from_numpy(L0), 0).double().to(self.device)
        X0 = torch.unsqueeze(torch.from_numpy(X0), 0).double().to(self.device)
        flux_0 = torch.ones(self.model.nb_frame - 1).double().to(self.device)
        fluxR_0 = torch.ones(self.model.nb_frame - 1).double().to(self.device)
        ref_amp_0 = torch.Tensor([1])

        fluxR_0[self.nb_ref:] = 0

        # ______________________________________
        #  Init variables and optimizer

        Lk, Xk = L0.clone(), X0.clone()
        flux_k, fluxR_k, ref_amp_k, k = flux_0.clone(), fluxR_0.clone(), ref_amp_0.clone(), 0
        Lk.requires_grad = True
        Xk.requires_grad = True

        # Loss and regularization at initialization 
        with torch.no_grad():

            Y0 = self.model.forward(L0, X0, flux_0, fluxR_0) if w_way[0] else 0
            Y0_reverse = self.model.forward_ADI_reverse(L0, X0, flux_0, fluxR_0) if w_way[1] else 0

            loss0 = w_way[0] * torch.sum(self.weight * (Y0 - science_data) ** 2) + \
                    w_way[1] * torch.sum(self.weight * (Y0_reverse - science_data_derot) ** 2)

            if w_pcent and Ractiv:  # Auto hyperparameters
                reg1 = self.regul1(X0, L0)
                w_r = w_rp[0] * loss0 / reg1 if w_rp[0] and reg1 > 0 else 0

                reg2 = self.regul2(X0, L0, self.mask, ref_amp_k)
                w_r2 = w_rp[1] * loss0 / reg2 if w_rp[1] and reg2 > 0 else 0
                
                reg3 = self.regul3(X0, L0)
                w_r3 = w_rp[2] * loss0 / reg3 if w_rp[0] and reg3 > 0 else 0

                if (w_rp[0] and reg1 < 0) or (w_rp[1] and reg2 < 0):
                    if verbose: print("Impossible to compute regularization weight. "
                                      "Activation is set to iteration n°2. ")
                    kactiv = 2

            R1_0 =  w_r * self.regul1(X0, L0) if w_r * Ractiv else 0
            R2_0 = Ractiv * w_r2 * self.regul2(X0, L0, self.mask, ref_amp_0)
            R3_0 = w_r3 * self.regul3(X0, L0) if w_r * Ractiv else 0

            loss0 += (R1_0 + R2_0 + R3_0)
        
        # Init  minimization terms
        loss, R1, R2, R3 = loss0, R1_0, R2_0, R3_0

        # Starting minization soon ...
        stat_msg = init_msg.format(self.name, save, *self.config, w_r, w_r2, str(maxiter))
        if verbose: print(stat_msg)
        txt_msg = stat_msg

        # ____________________________________
        # Nested functions

        # Definition of minimizer step.
        def closure():
            nonlocal R1, R2, R3, loss, w_r, w_r2, w_r3, Lk, Xk, flux_k, fluxR_k, ref_amp_k
            optimizer.zero_grad()  # Reset gradients

            # Background flux managment (beta)
            # bkg = self.compute_bkg(Xk[0])
            # with torch.no_grad() : Xk = Xk*self.coro
            # Lkb = Lk + bkg

            # Compute model(s)
            Yk = self.model.forward(Lk, Xk, flux_k, fluxR_k) if w_way[0] else 0
            Yk_reverse = self.model.forward_ADI_reverse(Lk, Xk, flux_k, fluxR_k) if w_way[1] else 0

            # Compute regularization(s)
            R1 = Ractiv * w_r  * self.regul1(Xk, Lk) if Ractiv * w_r else 0
            R2 = Ractiv * w_r2 * self.regul2(Xk, Lk, self.mask, ref_amp_k) if Ractiv * w_r2 else 0
            R3 = Ractiv * w_r3 * self.regul3(Xk, Lk) if Ractiv * w_r3 else 0

            # Compute loss and local gradients
            loss = w_way[0] * torch.sum(self.weight * (Yk - science_data) ** 2) + \
                   w_way[1] * torch.sum(self.weight * (Yk_reverse - science_data_derot) ** 2) + \
                   (R1 + R2 + R3)

            loss.backward()
            return loss

        # Definition of regularization activation
        def activation():
            nonlocal w_r, w_r2, w_r3, optimizer, Xk, Lk, flux_k, fluxR_k, ref_amp_k
            for activ_step in ["ACTIVATED", "AJUSTED"]:  # Activation in two step

                # Second step : re-compute regul after performing a optimizer step
                if activ_step == "AJUSTED": optimizer.step(closure)

                with torch.no_grad():
                    if w_pcent:
                        reg1 = self.regul1(Xk, Lk)
                        w_r = w_rp[0] * loss / reg1 if w_rp[0] and reg1 > 0 else 0

                        reg2 = self.regul2(Xk, Lk, self.mask, ref_amp_k)
                        w_r2 = w_rp[1] * loss / reg2 if w_rp[1] and reg2 > 0 else 0
                        
                        reg3 = self.regul3(Xk, Lk)
                        w_r3 = w_rp[2] * loss / reg3 if w_rp[0] and reg1 > 0 else 0

                    if (w_rp[0] and reg1 < 0) or (w_rp[1] and reg2 < 0) or (w_rp[2] and reg3 < 0):
                        if verbose: print("Impossible to compute regularization weight. Set to 0")

                # Define the varaible to be estimated.
                if estimI == "Both":
                    optimizer = optim.LBFGS([Lk, Xk, flux_k, fluxR_k])
                    flux_k.requires_grad = True;
                    fluxR_k.requires_grad = True
                elif estimI == "Frame":
                    optimizer = optim.LBFGS([Lk, Xk, flux_k])
                    flux_k.requires_grad = True
                elif estimI == "L":
                    optimizer = optim.LBFGS([Lk, Xk, fluxR_k])
                    fluxR_k.requires_grad = True
                elif estimI == "ref":
                    optimizer = optim.LBFGS([Lk, Xk, ref_amp_k])
                    ref_amp_k.requires_grad = True
                else:
                    optimizer = optim.LBFGS([Lk, Xk])

                if activ_step == "AJUSTED":
                    process_to_prints(activ_msg.format(activ_step, w_r, w_r2), -0.5)

        # Definition of print routines
        def process_to_prints(extra_msg=None, sub_iter=0.0, last=False):
            nonlocal txt_msg
            k_print = int(k) if sub_iter == 0 else k + sub_iter
            iter_msg = info_iter.format(k_print, *loss_ratio(Ractiv, float(R1), float(R2), float(loss)), loss)
            if extra_msg is not None: txt_msg += "\n" + extra_msg
            txt_msg += "\n" + iter_msg
            est_info = stat_msg.split("\n", 4)[3] + '\n'
            if gif: print_iter(Lk, Xk, flux_k, k + sub_iter, est_info + iter_msg, extra_msg, save, self.coro)
            if verbose: print(iter_msg, end=overwrite if not last else "\n\n")

        # Define the varaible to be estimated.
        if estimI == "Both":
            optimizer = optim.LBFGS([Lk, Xk, flux_k, fluxR_k])
            flux_k.requires_grad = True
            fluxR_k.requires_grad = True
        elif estimI == "Frame":
            optimizer = optim.LBFGS([Lk, Xk, flux_k])
            flux_k.requires_grad = True
        elif estimI == "L":
            optimizer = optim.LBFGS([Lk, Xk, fluxR_k])
            fluxR_k.requires_grad = True
        elif estimI == "ref":
            optimizer = optim.LBFGS([Lk, Xk, ref_amp_k])
            ref_amp_k.requires_grad = True
        else:
            optimizer = optim.LBFGS([Lk, Xk])

        # Save & prints the first iteration
        loss_evo.append(loss)
        self.last_iter = (L0, X0, flux_0, fluxR_k) if estimI else (L0, X0)
        if verbose: print(head)
        txt_msg += "\n" + head
        process_to_prints()

        start_time = datetime.now()

        # ____________________________________
        # ------ Minimization Loop ! ------ #

        try:
            for k in range(1, maxiter + 1):

                # Activation
                if kactiv and k == kactiv:
                    Ractiv = 1;
                    mink = k + 2
                    activation()
                    continue

                # Descactivation
                if kdactiv and k == kdactiv: Ractiv = 0

                # -- MINIMIZER STEP -- #
                optimizer.step(closure)
                if k > 1 and (torch.isnan(loss) or loss > loss_evo[-1]): self.final_estim = deepcopy(self.last_iter)

                # Save & prints
                loss_evo.append(loss)

                self.last_iter = (Lk, Xk, flux_k, fluxR_k) if estimI else (Lk, Xk)
                if k == 1: self.first_iter = (Lk, Xk, flux_k, fluxR_k) if estimI else (Lk, Xk)

                # Break point (based on gtol)
                grad = abs(loss[-1] - loss_evo[-2])
                
                if k > mink and  (grad < gtol) :
                    if not Ractiv and kactiv:  # If regul haven't been activated yet, continue with regul
                        Ractiv = 1;
                        mink = k + 2;
                        kactiv = k
                        activation()
                    else:
                        ending = 'max iter reached' if k == maxiter else 'Gtol reached'
                        break
                elif torch.isnan(loss):  # Also break if an error occure.
                    ending = 'Nan values end minimization. Last value estimated will be returned.'
                    break

                process_to_prints()

        except KeyboardInterrupt:
            ending = "keyboard interruption"


        # ______________________________________
        # Done, store and unwrap results back to numpy array!

        process_to_prints(last=True)
        end_msg = "Done (" + ending + ") - Running time : " + str(datetime.now() - start_time)
        if verbose: print(end_msg)
        txt_msg += "\n\n" + end_msg

        if k > 1 and (torch.isnan(loss) or loss > loss_evo[-2]) and self.final_estim is not None:
            L_est = abs(self.final_estim[0].detach().numpy()[0])
            X_est = abs(self.final_estim[1].detach().numpy()[0])
        else:
            L_est, X_est = abs(Lk.detach().numpy()[0]), abs(Xk.detach().numpy()[0])

        flux = abs(flux_k.detach().numpy())
        fluxR = abs(fluxR_k.detach().numpy())
        amp_ref = abs(ref_amp_k.detach().numpy())
        loss_evo = [float(lossk.detach().numpy()) for lossk in loss_evo]

        # Remove bkg flux from bkg_pup. (beta)
        bkg = 0  # np.median(X_est[np.where(self.pup_bkg==1)])

        nice_L_est = self.coro.numpy() * (L_est + bkg).clip(min=0)
        nice_X_est = self.coro.numpy() * (X_est - bkg).clip(min=0)

        # Result dict
        res = {'state': optimizer.state,
               'x': (nice_L_est, nice_X_est),
               'x_no_r': (L_est, X_est),
               'flux': flux,
               'amp_ref': amp_ref,
               'fluxR': fluxR,
               'loss_evo': loss_evo,
               'Kactiv': kactiv,
               'ending': ending}

        self.res = res

        # Save
        if gif: iter_to_gif(save, self.name)

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)

            write_fits(self.savedir + "/L_est" + self.name, nice_L_est)
            write_fits(self.savedir + "/X_est" + self.name, nice_X_est)
            # write_fits(self.savedir + "/X_est"+self.name, nice_X_est)

            with open(self.savedir + "/config.txt", 'w') as f:
                f.write(txt_msg)

            if estimI:
                write_fits(self.savedir + "/flux" + self.name, flux)
                write_fits(self.savedir + "/fluxR" + self.name, fluxR)

        if estimI:
            return nice_L_est, nice_X_est, flux
        else:
            return nice_L_est, nice_X_est

    #### SETTERS / CONFIG ####
    
    def set_savedir(self, savedir: str):
        self.savedir = savedir
        if not isdir(self.savedir): makedirs(self.savedir)

    def set_init(self, X0=None, L0=None):
        """
        Define initialization by yourslef.

        Parameters
        ----------
        X0 : numpy.ndarry or None
            Init of circumstellar map
        L0 : numpy.ndarry or None
            Init of speakles map

        """
        if L0 is None and X0 is None:
            raise (AssertionError("At least one argument must be provided"))
            
        if X0 is None:
            X0 = np.mean(cube_derotate(self.science_data - L0, self.model.rot_angles), 0)
        elif L0 is None:
            L0 = np.mean(self.science_data -
                         cube_derotate(np.tile(X0, (self.nb_frame, 1, 1)), -self.model.rot_angles), 0)

        self.L0x0 = (L0.clip(0), X0.clip(0))

    def configR1(self, mode: str, smoothL=True, p_L=1, p_X=1, epsi=1e-7):
        """ Configuration of first regularization. (smooth-like)"""

        if mode == "smooth_with_edges":
            self.smooth = lambda X: torch.sum(self.coroR * sobel_tensor_conv(X, axis='y') ** 2 - epsi ** 2) + \
                                    torch.sum(self.coroR * sobel_tensor_conv(X, axis='x') ** 2 - epsi ** 2)
        elif mode == "smooth":
            self.smooth = lambda X: torch.sum(sobel_tensor_conv(X, axis='y') ** 2) + \
                                    torch.sum(sobel_tensor_conv(X, axis='x') ** 2)

        elif mode == "l1":
            self.smooth = lambda X: torch.sum(self.coroR * torch.abs(X))

        self.p_L = p_L
        self.p_X = p_X

        if smoothL:
            self.regul1 = lambda X, L: self.p_X * self.smooth(X) + self.p_L * self.smooth(L)
        else:
            self.regul1 = lambda X, L: self.smooth(X)


    def configR2(self, Msk=None, mode="mask", penaliz="X", invert=False, save=False):
        """ Configuration for Second regularization.
        Two possible mode :   R = M*X      (mode 'mask')
                           or R = dist(M-X) (mode 'dist')
                           or R = sum(X)   (mode 'l1')

        Parameters
        ----------
        save : False
            Save your mask

        Msk : numpy.array or torch.tensor
            Prior mask of X.

        mode : {"mask","dist"}.
            if "mask" :  R = norm(M*X)
                if M will be normalize to have values from 0 to 1.
            if "dist" :  R = dist(M-X)

        penaliz : {"X","L","both"}
            Unknown to penaliz.
            With mode mask :
                if "X" : R = norm(M*X)
                if "L" : R = norm((1-M)*L)
                if "B" : R = norm(M*X) + norm((1-M)*L)
            With mode dist :
                if "X" : R = dist(M-X)
                if "L" : R = dist(M-L)
                if "B" : R = dist(M-X) - dist(M-L)
            With mode l1 :
                if "X" : R = sum(X)
                if "L" : R = sum(L)
                if "B" : R = sum(X) - sum(L)

        invert : Bool
            Reverse penalize mode bewteen L and X.

        Returns
        -------

        """
        if mode == "pdi":
            Msk = convert_to_mask(Msk)
            mode = "mask"

        if mode != 'l1':
            if isinstance(Msk, np.ndarray): Msk = torch.from_numpy(Msk)
            if not (isinstance(Msk, torch.Tensor) and Msk.shape == self.model.frame_shape):
                raise TypeError("Mask M should be tensor or arr y of size " + str(self.model.frame_shape))

        if Msk is not None and (mode != 'mask' and mode != 'ref'):
            warnings.warn(UserWarning("You provided a mask but did not chose 'mask' option"))

        self.F_rp = create_radial_prof_matirx(self.model.frame_shape)  # Radial profil transform
        # y, x = np.indices(self.model.frame_shape)
        # self.yx = torch.from_numpy(y), torch.from_numpy(x)

        penaliz = penaliz.capitalize()
        rM = self.coroR  # corono mask for regul
        if penaliz not in ("X", "L", "Both", "B"):
            raise Exception("Unknown value of penaliz. Possible values are {'X','L','B'}")

        if mode == "dist":
            self.mask = Msk
            sign = -1 if invert else 1
            if penaliz == "X":
                self.regul2 = lambda X, L, M: tsum(rM * (M - X) ** 2)
            elif penaliz == "L":
                self.regul2 = lambda X, L, M: tsum(rM * (M - L) ** 2)
            elif penaliz in ("Both", "B"):
                self.regul2 = lambda X, L, M: sign * (tsum(rM * (M - X) ** 2) -
                                                      tsum(rM * (M - L) ** 2))

        elif mode == "mask":
            Msk = Msk / torch.max(Msk)  # Normalize mask
            self.mask = (1 - Msk) if invert else Msk
            if penaliz == "X":
                self.regul2 = lambda X, L, M, amp: tsum(rM * (M * X) ** 2)
            elif penaliz == "L":
                self.regul2 = lambda X, L, M, amp: tsum(rM * ((1 - M) * L) ** 2)
            elif penaliz in ("Both", "B"):
                self.regul2 = lambda X, L, M, amp: tsum(rM * (M * X) ** 2) + \
                                                   tsum(rM * M) ** 2 / tsum(rM * (1 - M)) ** 2 * \
                                                   tsum(rM * ((1 - M) * L) ** 2) + \
                                                   res_non_convexe(radial_profil(L, self.F_rp))

        elif mode == "l1":
            sign = -1 if invert else 1
            if penaliz == "X":
                self.regul2 = lambda X, L, M: tsum(X ** 2)
            elif penaliz == "L":
                self.regul2 = lambda X, L, M: tsum(L ** 2)
            elif penaliz in ("Both", "B"):
                self.regul2 = lambda X, L, M: sign * (tsum(X ** 2) - tsum(L ** 2))

        elif mode == "ref":
            self.mask = np.mean(cube_derotate(np.tile(Msk, (self.nb_frame, 1, 1)), self.model.rot_angles), axis=0)
            self.mask = radial_profil(self.coro * torch.Tensor(self.mask), self.F_rp)
            self.r2 = torch.linspace(0, len(self.mask), len(self.mask)) ** 2
            self.regul2 = lambda X, L, M, amp: tsum((amp * M - radial_profil(self.coro * torch.mean(
                self.model.get_Lf(L, rot=True), dim=0)[0], self.F_rp)) ** 2)

        else:
            raise Exception("Unknown value of mode. Possible values are {'mask','dist','l1'}")

        R2_name = mode + " on " + penaliz
        R2_name += " inverted" if invert else ""
        self.config[1] = R2_name

        if save and mode != "l1":
            if not isdir(self.savedir): makedirs(self.savedir)
            write_fits(self.savedir + "/maskR2" + self.name, self.mask.numpy())

        def configR3(self, mode: str, smoothL=True, p_L=1, p_X=1, epsi=1e-7):
                """ Configuration of first regularization. (smooth-like)"""
        
                if mode == "smooth_with_edges":
                    self.smooth = lambda X: torch.sum(self.coroR * sobel_tensor_conv(X, axis='y') ** 2 - epsi ** 2) + \
                                            torch.sum(self.coroR * sobel_tensor_conv(X, axis='x') ** 2 - epsi ** 2)
                elif mode == "smooth":
                    self.smooth = lambda X: torch.sum(sobel_tensor_conv(X, axis='y') ** 2) + \
                                            torch.sum(sobel_tensor_conv(X, axis='x') ** 2)
        
                elif mode == "l1":
                    self.smooth = lambda X: torch.sum(self.coroR * torch.abs(X))
        
                else:
                    self.smooth = lambda X: 0
        
                self.p_L = p_L
                self.p_X = p_X
        
                if smoothL:
                    self.regul3 = lambda X, L: self.p_X * self.smooth(X) + self.p_L * self.smooth(L)
                else:
                    self.regul3 = lambda X, L: self.smooth(X)

    
    #### GETTERS / PLOTS ####

    def get_science_data(self):
        """Return input cube and angles"""
        return self.model.rot_angles, self.science_data

    def get_residual(self, way="direct", save=False):
        """Return input cube and angles"""

        if way == "direct":
            science_data = torch.unsqueeze(torch.from_numpy(self.science_data), 1).double().to(self.device)
            reconstructed_cube = self.model.forward(*self.last_iter)  # Reconstruction on last iteration
            residual_cube = science_data - reconstructed_cube

        elif way == "reverse":
            science_data_derot_np = cube_derotate(self.science_data, self.model.rot_angles)
            science_data = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 1).double().to(self.device)
            reconstructed_cube = self.model.forward_ADI_reverse(*self.last_iter)  # Reconstruction on last iteration
            residual_cube = science_data - reconstructed_cube

        else:
            raise (ValueError, "way sould be 'reverse' or 'direct'")

        nice_residual = self.coro.detach().numpy() * residual_cube.detach().numpy()[:, 0, :, :]
        if save:
            savedir = save if isinstance(save, str) else self.savedir
            if not isdir(savedir): makedirs(savedir)
            write_fits(savedir + "/residual_" + way + "_" + self.name, nice_residual)

        return nice_residual

    def get_evo_convergence(self, show=True, save=False):
        """Return loss evolution"""

        loss_evo = self.res['loss_evo']
        Kactiv = self.res["Kactiv"] + 1 if isinstance(self.res["Kactiv"], (int, float)) else len(loss_evo) - 1

        if show:
            plt.ion()
        else:
            plt.ioff()

        plt.close("Evolution of loss criteria")
        fig = plt.figure("Evolution of loss criteria", figsize=(16, 9))
        plt.cla(), plt.cla()
        fig.subplots(1, 2, gridspec_kw={'width_ratios': [Kactiv, len(loss_evo) - Kactiv]})

        plt.subplot(121), plt.xlabel("Iteration"), plt.ylabel("Loss - log scale"), plt.yscale('log')
        plt.plot(loss_evo[:Kactiv], 'X-', color="tab:orange"), plt.title("Loss evolution BEFORE activation")

        plt.subplot(122), plt.xlabel("Iteration"), plt.ylabel("Loss - log scale"), plt.yscale('log')
        plt.plot(loss_evo[Kactiv:], 'X-', color="tab:blue"), plt.title("Loss evolution, AFTER activation")
        plt.xticks(range(len(loss_evo) - Kactiv), range(Kactiv, len(loss_evo)))

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            plt.savefig(self.savedir + "/convergence_" + self.name)

        return loss_evo

    def get_speckles(self, show=True, save=False):

        if self.speckles is None:
            res = np.min(self.science_data, 0)
            self.ambiguities = np.min(cube_derotate(
                np.tile(res, (self.nb_frame, 1, 1)), self.model.rot_angles), 0)
            self.speckles = res - self.ambiguities

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            write_fits(self.savedir + "/speckles" + self.name, self.speckles)
            write_fits(self.savedir + "/speckles_and_stellar_halo" + self.name, self.speckles + self.ambiguities)

        if show:
            plt.figure("Speckles map", figsize=(16, 9))
            plt.imshow(self.speckles, cmap='jet')
            plt.show()

        return self.speckles

    def get_result_unrotated(self):
        return self.res["x"]

    def get_ambiguity(self, show=True, save=False):

        if self.ambiguities is None:
            res = np.min(self.science_data, 0)
            self.ambiguities = np.min(cube_derotate(np.tile(res, (self.nb_frame, 1, 1)), -self.model.rot_angles), 0)

            self.stellar_halo = np.min(cube_derotate(np.tile(self.ambiguities, (50, 1, 1)),
                                                     list(np.linspace(0, 360, 50))), 0)

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            write_fits(self.savedir + "/ambiguities_halo" + self.name, self.ambiguities)
            write_fits(self.savedir + "/stellar_halo" + self.name, self.stellar_halo)
            write_fits(self.savedir + "/ambiguities" + self.name, self.ambiguities - self.stellar_halo)

        if show:
            plt.figure("Ambiguities", figsize=(16, 9))
            plt.imshow(self.ambiguities, cmap='jet')

        return self.ambiguities

    def get_rot_weight(self, show=True, save=False):
        """Return loss evolution"""

        weight = self.ang_weight.detach().numpy()[:, 0, 0, 0]
        rot_angles = self.model.rot_angles

        if show:
            plt.ion()
        else:
            plt.ioff()

        plt.figure("Frame weight based on PA", figsize=(16, 9))

        plt.subplot(211), plt.xlabel("Frame"), plt.ylabel("PA in deg")
        plt.plot(rot_angles, 'X-', color="tab:purple"), plt.title("Angle for each frame")

        plt.subplot(212), plt.bar(range(len(weight)), weight, color="tab:cyan", edgecolor="black")
        plt.title("Assigned weight"), plt.xlabel("Frame"), plt.ylabel("Frame weight")
        if show: plt.show()

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            plt.savefig(self.savedir + "/PA_frame_weight_" + self.name)

        return weight
   
    def get_radial_prof(self, show=True, save=False):

        if show:
            plt.ion()
        else:
            plt.ioff()

        L, X = self.res["x_no_r"]
        amp_ref = self.res['amp_ref']

        L = torch.Tensor(L)
        Rl = radial_profil(self.coro * L, self.F_rp).numpy()
        Rpsf = self.mask.numpy() * amp_ref

        plt.figure("Radial profile")

        plt.plot(Rl, label="Estimated speckles field", lw=2)
        plt.plot(Rpsf, label="Reference radial profil", ls='--')
        plt.legend()

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            plt.savefig(self.savedir + "/radial_profil" + self.name)

        return Rl

    def get_flux(self, show=True, save=False):
        """Return relative flux variations between frame"""

        flux = self.res['flux']
        fluxR = self.res['fluxR']

        if show:
            plt.ion()
        else:
            plt.ioff()

        plt.figure("Relative flux variations between frame", figsize=(16, 9))
        lim = max(abs((flux - 1)))
        limR = max(abs((fluxR - 1)))
        if lim == 0: lim += 1
        if limR == 0: limR += 1

        plt.subplot(1, 2, 1), plt.bar(range(len(flux)), flux - 1, bottom=1, color='tab:red', edgecolor="black")
        plt.ylabel("Flux variation"), plt.xlabel("Frame"), plt.title("Flux variations between Frames")
        plt.ylim([1 - lim, 1 + lim]), plt.ticklabel_format(useOffset=False)

        plt.subplot(1, 2, 2), plt.bar(range(len(fluxR)), fluxR - 1, bottom=1, color='tab:green', edgecolor="black")
        plt.ylabel("Flux variation"), plt.xlabel("Frame"), plt.title("Flux variations of starlight map")
        plt.ylim([1 - limR, 1 + limR]), plt.ticklabel_format(useOffset=False)
        if show: plt.show()

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            plt.savefig(self.savedir + "/flux_" + self.name)

        return flux, fluxR

    def get_cube_without_speckles(self, way="direct", save=False):
        """Return input cube and angles"""

        Lk, _, flux_k, fluxR_k = self.last_iter
        if way == "direct":
            science_data = torch.unsqueeze(torch.from_numpy(self.science_data), 1).double().to(self.device)
            reconstructed_cube = science_data - self.model.get_Lf(Lk, flux_k, fluxR_k)

        elif way == "reverse":
            science_data_derot_np = cube_derotate(self.science_data, self.model.rot_angles)
            science_data = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 1).double().to(self.device)
            reconstructed_cube = science_data - self.model.get_Rx(Lk, flux_k, fluxR_k, inverse=True)

        else:
            raise (ValueError, "way sould be 'reverse' or 'direct'")

        reconstructed_cube = reconstructed_cube.detach().numpy()[:, 0, :, :]
        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            write_fits(self.savedir + "/cube_without_speckles_" + way + "_" + self.name, reconstructed_cube)

        return reconstructed_cube

    def get_reconstruction(self, way="direct", save=False):
        """Return input cube and angles"""

        if way == "direct":
            reconstructed_cube = self.model.forward(*self.last_iter)  # Reconstruction on last iteration

        elif way == "reverse":
            reconstructed_cube = self.model.forward_ADI_reverse(*self.last_iter)  # Reconstruction on last iteration

        else:
            raise (ValueError, "way sould be 'reverse' or 'direct'")

        reconstructed_cube = self.coro.numpy() * reconstructed_cube.detach().numpy()[:, 0, :, :]
        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            write_fits(self.savedir + "/reconstruction_" + way + "_" + self.name, reconstructed_cube)

        return reconstructed_cube

    def get_initialisation(self, save=False):
        """Return input cube and angles"""

        if self.L0x0 is None: raise "No initialisation have been performed"
        L0, X0 = self.L0x0

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            if not isdir(self.savedir + "/L0X0/"): makedirs(self.savedir + "/L0X0/")
            print("Save init from in " + self.savedir + "/L0X0" + "...")

            bkg = np.median(X0[np.where(self.pup_bkg == 1)])
            nice_X0 = self.coro.numpy() * (X0 - bkg).clip(min=0)
            L0 = L0 + bkg

            write_fits(self.savedir + "/L0X0/" + "/L0.fits", L0, verbose=False)
            write_fits(self.savedir + "/L0X0/" + "/X0.fits", X0, verbose=False)
            write_fits(self.savedir + "/L0X0/" + "/nice_X0.fits", nice_X0, verbose=False)

        return L0, X0

    def mustard_results(self, per_vmax=99, r_no_scale=False):
       """Return loss evolution"""
    
       L, X = self.res["x_no_r"]
       cube = self.science_data
       ang = self.model.rot_angles
    
       noise = self.coro.numpy() * self.get_residual()
       flx, flxR = self.get_flux(show=False)
       flx = [1] + list(flx)
       flxR = [1] + list(flxR)
    
       font = {'color': 'white',
               'weight': 'bold',
               'size': 16,
               }
    
       vmax = np.percentile(cube, per_vmax)
       vmin = cube.min()
    
       Rvmax = np.percentile(X, per_vmax) if r_no_scale else vmax
    
       def plot_framek(val: int, show=True) -> None:
    
           num = int(val)
           font["size"] = 26
           font["color"] = "white"
    
           plt.subplot(1, 2, 1)
           plt.imshow(self.coro.numpy() * cube[num], vmax=vmax, vmin=vmin, cmap='jet')
           plt.text(20, 40, "ADI cube", font)
           plt.title("Frame n°" + str(num))
           font["size"] = 22
           plt.text(20, 55, r'$\Delta$ Flux : 1{:+.2e}'.format(1 - flx[num]), font)
    
           font["size"] = 22
           plt.subplot(2, 2, 4)
           plt.imshow(self.final_mask * flx[num] * frame_rotate(X, ang[num]), vmax=Rvmax, vmin=vmin, cmap='jet')
           plt.text(20, 40, "Rotate", font)
    
           font["size"] = 16
           plt.subplot(2, 4, 4)
           plt.imshow(self.final_mask * flxR[num] * flx[num] * L, vmax=vmax, vmin=vmin, cmap='jet')
           plt.text(20, 40, "Static", font)
           font["size"] = 12
           plt.text(20, 55, r'$\Delta$ Flux : 1{:+.2e}'.format(1 - flxR[num]), font)
    
           font["size"] = 16
           font["color"] = "red"
    
           plt.subplot(2, 4, 3)
           plt.imshow(noise[num], cmap='jet')
           plt.clim(-np.percentile(noise[num], 98), +np.percentile(noise[num], 98))
           plt.text(20, 40, "Random", font)
           if show: plt.show()
    
       # ax_slid = plt.axes([0.1, 0.25, 0.0225, 0.63])
       # handler = Slider(ax=ax_slid, label="Frame", valmin=0, valmax=len(cube), valinit=0, orientation="vertical")
       # handler.on_changed(plot_framek)
    
       plt.ioff()
       plt.figure("TMP_MUSTARD", figsize=(16, 14))
       if not isdir(self.savedir): makedirs(self.savedir)
       if not isdir(self.savedir + "/tmp/"): makedirs(self.savedir + "/tmp/")
    
       images = []
       for num in range(len(cube)):
           plt.cla();
           plt.clf()
           plot_framek(num, show=False)
           plt.savefig(self.savedir + "/tmp/noise_" + str(num) + ".png")
    
       for num in range(len(cube)):
           images.append(Image.open(self.savedir + "/tmp/noise_" + str(num) + ".png"))
    
       for num in range(len(cube)):
           try:
               remove(self.savedir + "/tmp/noise_" + str(num) + ".png")
           except Exception as e:
               print("[WARNING] Failed to delete iter .png : " + str(e))
    
       try:
           rmdir(self.savedir + "/tmp/")
       except Exception as e:
           print("[WARNING] Failed to remove iter dir : " + str(e))
    
       images[0].save(fp=self.savedir + "MUSTARD.gif", format='GIF',
                      append_images=images, save_all=True, duration=200, loop=0)
    
       plt.close("TMP_MUSTARD")
       plt.ion()

# %% -----Small util function ----------------------------------------------------
# Could be moved to utils / algos module...

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

            
def cube_rotate(cube, angles):
    new_cube = torch.zeros(cube.shape)
    for ii in range(len(angles)):
        new_cube[ii] = rotate(torch.unsqueeze(cube[ii], 0), float(angles[ii]),
                              InterpolationMode.BILINEAR)[0]
    return new_cube

def normlizangle(angles: np.array) -> np.array:
    """Normaliz an angle between 0 and 360"""

    angles[angles < 0] += 360
    angles = angles % 360

    return angles

def compute_rot_weight(angs: np.array) -> np.array:
    nb_frm = len(angs)

    # Set value of the delta offset max
    max_rot = np.mean(abs(angs[:-1] - angs[1:]))
    # if max_rot > 1 : max_rot = 1
    max_rot = 100

    ang_weight = []
    for id_ang, ang in enumerate(angs):
        # If a frame neighbours delta-rot under max_rot, frames are less valuable than other frame (weight<1)
        # If the max_rot is exceed, the frame is not consider not more valuable than other frame (weight=1)
        # We do this for both direction's neighbourhoods

        nb_neighbours = 0;
        detla_ang = 0
        tmp_id = id_ang

        while detla_ang < max_rot and tmp_id < nb_frm - 1:  # neighbours after
            tmp_id += 1;
            nb_neighbours += 1
            detla_ang += abs(angs[tmp_id] - angs[id_ang])
        w_1 = (1 + abs(angs[id_ang] - angs[id_ang + 1])) / nb_neighbours if nb_neighbours > 1 else 1

        tmp_id = id_ang;
        detla_ang = 0
        nb_neighbours = 0
        while detla_ang < max_rot and tmp_id > 0:  # neighbours before
            tmp_id -= 1;
            nb_neighbours += 1
            detla_ang += abs(angs[tmp_id] - angs[id_ang])
        w_2 = (1 + abs(angs[id_ang] - angs[id_ang - 1])) / nb_neighbours if nb_neighbours > 1 else 1

        ang_weight.append((w_2 + w_1) / 2)

    return np.array(ang_weight)
