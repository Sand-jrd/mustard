#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:30:10 2021

______________________________

|      Neo-Mayo estimator     |
______________________________


@author: sand-jrd
"""

# For file management
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import cube_derotate, frame_rotate

from os import makedirs, remove, rmdir
from os.path import isdir

# Algo and science model
from mustard.algo import init_estimate, sobel_tensor_conv, convert_to_mask
import torch
import numpy as np


# Loss functions
import torch.optim as optim
from torch import sum as tsum
from torch.nn import ReLU as relu_constr

# Other
from mustard.utils import circle, iter_to_gif, print_iter
from mustard.model import model_ADI, model_ASDI
from copy import deepcopy

# -- For verbose -- #
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
from PIL import Image

sep = ('_' * 50)
head = "|it |       loss        |        R1        |        R2        |       Rpos       |       total      |"
info_iter = "|{:^3}|{:.6e} ({:^3.0f}%)|{:.6e} ({:^2.0f}%)|{:.6e} ({:^2.0f}%)|{:.6e} ({:^2.0f}%)|{:.12e}|"

init_msg = sep + "\nResolving IP-ADI optimization problem - name : {}" +\
                 "\n Outputs will be saved {}" +\
                 "\nRegul R1 : '{}' and R2 : '{}'" +\
                 "\n{} deconvolution and {} frame weighted based on rotations" + \
                 "\nRelative amplitude of {} will be estimated" + \
                 "\nRegul weight are set to w_r={:.2f}% and w_r2={:.2f}%, maxiter={}\n"

activ_msg = "REGUL HAVE BEEN {} with w_r={:.2f} and w_r2={:.2f}"

ReLU = relu_constr()

def loss_ratio(Ractiv: int or bool, R1: float, R2: float, Rp: float, L: float) -> tuple:
    """ Compute Regul weight over data attachment terme """
    return tuple(np.array((1, 100/L)) * (L - Ractiv * (abs(R1) + abs(R2)) - abs(Rp) )) + \
           tuple(np.array((1, 100/L)) * abs(R1)) + \
           tuple(np.array((1, 100/L)) * abs(R2)) + \
           tuple(np.array((1, 100/L)) * abs(Rp))



# %% ------------------------------------------------------------------------
class mustard_estimator:
    """ Neo-mayo Algorithm main class  """

    def __init__(self, science_data: np.ndarray, angles: np.ndarray, scale=None, coro=6, pupil="edge",
                 psf=None, Badframes=None, savedir='./'):
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

        pupil : int, None or "edge"
            Size of the pupil.
            If pupil is set to "edge" : the pupil raduis will be half the size of the frame
            If pupil is set to None : there will be no pupil at all

        psf : np.ndarray or None
            If a psf is provided, will perform deconvolution (i.e conv by psf inculded in forward model)
            /!\ "psf" key must be added in json import file

        Badframes : tuple or list or None
            Bad frames that you will not be taken into account

        savedir : str
            Path for outputs
        """

        # -- Create model and define constants --

        if Badframes is not None:
            science_data = np.delete(science_data, Badframes, 0)
            angles = np.delete(angles, Badframes, 0)
            scale = np.delete(scale, Badframes, 0)


        # Constants
        self.shape = science_data[0].shape
        self.nb_frame = science_data.shape[0]
        self.L0x0 = None
        self.mask = 1  # R2 mask default. Will be set if call R2 config
        self.name = ''
        self.savedir = savedir

        # Coro and pupil masks
        pupilR = pupil
        if pupil is None: pupil = np.ones(self.shape)
        elif pupil == "edge":
            pupil  = circle(self.shape, self.shape[0]/2)
            pupilR = circle(self.shape, self.shape[0]/2 - 2)
        elif isinstance(pupil, (int, float)) :
            pupil  = circle(self.shape, pupil)
            pupilR = circle(self.shape, pupilR)
        else: raise ValueError("Invalid pupil key argument. Possible values : {float/int, None, 'edge'}")

        if coro is None: coro = 0
        self.coro   = (1 - circle(self.shape, coro)) * pupil
        self.coroR  = (1 - circle(self.shape, coro)) * pupilR

        # Convert to tensor
        rot_angles = normlizangle(angles)
        self.science_data = science_data
        self.model = model_ASDI(rot_angles, scale, self.coro, psf) if scale \
            else model_ADI(rot_angles, self.coro, psf)

        if self.model.nb_frame != self.model.nb_frame :
            raise("Length of angles/scales does not match the size of the science-data cube !")

        # Will be filled with weight if anf_weight option is activated
        self.ang_weight = torch.from_numpy(np.ones(self.nb_frame).reshape((self.nb_frame, 1, 1, 1))).double()


        self.coro  = torch.from_numpy(self.coro).double()
        self.coroR = torch.from_numpy(self.coroR).double()

        # -- Configure regularization (can be change later, this is defaults parameters)
        self.config = ["smooth", None, 'No' if psf is None else 'With','no', 'Both L and X']
        self.configR2(Msk=None, mode="l1", penaliz="X")
        self.configR1(mode="smooth")

        self.res = None; self.last_iter = None; self.first_iter = None; self.final_estim = None
        self.ambiguities = None; self.speckles = None; self.science_data_ori=None # init results vars

    def set_init(self, X0 = None, L0 = None):
        """
        Define initialization by yourslef.

        Parameters
        ----------
        X0 : numpy.ndarry or None
            Init of circumstellar map
        L0 : numpy.ndarry or None
            Init of speakles map

        Returns
        -------

        """
        if L0 is None and X0 is None:
            raise(AssertionError("At least one argument must be provided"))

        if X0 is None :
            X0 = np.min(cube_derotate(self.science_data - L0, self.model.rot_angles), 0)
        elif L0 is None :
            L0 = np.min(self.science_data -
                        cube_derotate(np.tile(X0, (self.nb_frame, 1, 1)), -self.model.rot_angles), 0)

        self.L0x0 = (L0, X0)

    def initialisation(self, from_dir=None, save=None, Imode="max_common",  **kwargs):
        """ Init L0 (starlight) and X0 (circonstellar) with a PCA
         
         Parameters
         ----------
          from_dir : str (optional)
             if None, compute PCA
             if a path is given, get L0 (starlight) and X0 (circonstellar) from the directory

          save : bool
             if a True, L0 (starlight) and X0 (circonstellar) will be saved at self.savedir

          Imode : str
            Type of initialisation : {'pcait','pca'}

          kwargs :
             Argument that will be pass to :
             vip_hci.pca.pca_fullfr.pca if 'Imode' = 'pca'
             vip_hci.itpca.pca_it if 'Imode' = 'pcait'

         Returns
         -------
         L_est, X_est: ndarray
             Estimated starlight (L) 
             and circunstlellar contributions (X) 
         
        """
        warnings.warn(DeprecationWarning("this will be removed in the future !!!!"))
        if from_dir:
            print("Get init from pre-processed datas...")
            L0 = open_fits(from_dir + "/L0.fits", verbose=False)
            X0 = open_fits(from_dir + "/X0.fits", verbose=False)

        else:

            # -- Iterative PCA for variable init -- #
            start_time = datetime.now()
            print(sep + "\nInitialisation  ...")

            L0, X0 = init_estimate(self.science_data, self.model.rot_angles, Imode=Imode , **kwargs)

            print("Done - running time : " + str(datetime.now() - start_time) + "\n" + sep)

            if save:
                if not isdir(self.savedir + "/L0X0/"): makedirs(self.savedir + "/L0X0/")
                print("Save init from in " + self.savedir + "/L0X0/" + "...")

                nice_X0 = self.coro.numpy() * X0
                nice_X0 = (nice_X0 - np.median(nice_X0)).clip(0)

                write_fits(self.savedir + "/L0X0/" + "/L0.fits", L0, verbose=False)
                write_fits(self.savedir + "/L0X0/" + "/X0.fits", X0, verbose=False)
                write_fits(self.savedir + "/L0X0/" + "/nice_X0.fits", self.coro.numpy()*nice_X0, verbose=False)

        # -- Define constantes
        self.L0x0 = (L0, X0)

        return L0, X0

    def configR1(self, mode: str, smoothL = True, p_L = 1, epsi = 1e-7):
        """ Configuration of first regularization. (smooth-like)"""

        if mode == "smooth_with_edges":
            self.smooth = lambda X: torch.sum(self.coroR * sobel_tensor_conv(X, axis='y') ** 2 - epsi ** 2) +\
                                   torch.sum(self.coroR * sobel_tensor_conv(X, axis='x') ** 2 - epsi  ** 2)
        elif mode == "smooth" :
            self.smooth = lambda X: torch.sum(self.coroR * sobel_tensor_conv(X, axis='y') ** 2) +\
                                   torch.sum(self.coroR * sobel_tensor_conv(X, axis='x') ** 2)

        elif mode == "peak_preservation" :
            self.xmax = 1
            self.fpeak = lambda X: torch.log(self.xmax - X/self.xmax)
            self.smooth = lambda X: torch.sum(self.fpeak(X) * self.coroR * sobel_tensor_conv(X, axis='y') ** 2) + \
                                   torch.sum(self.fpeak(X) * self.coroR * sobel_tensor_conv(X, axis='x') ** 2)
        elif mode == "l1":
            self.smooth = lambda X: torch.sum(self.coroR * torch.abs(X))

        self.p_L = p_L
        if smoothL : self.regul = lambda X, L: self.smooth(X) + self.p_L*self.smooth(L)
        else       : self.regul = lambda X, L: self.smooth(X)

    def configR2(self, Msk=None, mode = "mask", penaliz = "X", invert=False, save=True):
        """ Configuration for Second regularization.
        Two possible mode :   R = M*X      (mode 'mask')
                           or R = dist(M-X) (mode 'dist')
                           or R = sum(X)   (mode 'l1')

        Parameters
        ----------
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

        if mode != 'l1' :
            if isinstance(Msk, np.ndarray): Msk = torch.from_numpy(Msk)
            if not (isinstance(Msk, torch.Tensor) and Msk.shape == self.model.frame_shape):
                raise TypeError("Mask M should be tensor or arr y of size " + str(self.model.frame_shape))

        if Msk is not None and mode != 'mask' :
            warnings.warn(UserWarning("You provided a mask but did not chose 'mask' option"))

        penaliz = penaliz.capitalize()
        rM = self.coro  # corono mask for regul
        if penaliz not in ("X", "L", "Both", "B") :
            raise Exception("Unknown value of penaliz. Possible values are {'X','L','B'}")

        if mode == "dist":
            self.mask = Msk
            sign = -1 if invert else 1
            if   penaliz == "X"   :  self.regul2 = lambda X, L, M: tsum( rM * (M - X) ** 2)
            elif penaliz == "L"   :  self.regul2 = lambda X, L, M: tsum( rM * (M - X) ** 2)
            elif penaliz in ("Both", "B"):  self.regul2 = lambda X, L, M: sign * (tsum( rM * (M - X) ** 2) -
                                                                                  tsum( rM * (M - L) ** 2))

        elif mode == "mask" :
            Msk = Msk/torch.max(Msk)  # Normalize mask
            self.mask = (1-Msk) if invert else Msk
            if   penaliz == "X"   : self.regul2 = lambda X, L, M: tsum( rM * (M * X) ** 2)
            elif penaliz == "L"   : self.regul2 = lambda X, L, M: tsum( rM * ((1 - M) * L) ** 2)
            elif penaliz in ("Both", "B"): self.regul2 = lambda X, L, M: tsum( rM * (M * X) ** 2) +\
                                                                  tsum(M)**2/tsum((1 - M))**2 *\
                                                                  tsum( rM * ((1 - M) * L) ** 2)

        elif mode == "l1":
            sign = -1 if invert else 1
            if   penaliz == "X"   : self.regul2 = lambda X, L, M: tsum(X ** 2)
            elif penaliz == "L"   : self.regul2 = lambda X, L, M: tsum(L ** 2)
            elif penaliz in ("Both", "B"): self.regul2 = lambda X, L, M: sign * (tsum(X ** 2) - tsum(L ** 2))

        else: raise Exception("Unknown value of mode. Possible values are {'mask','dist','l1'}")

        R2_name = mode + " on " + penaliz
        R2_name += " inverted" if invert else ""
        self.config[1] = R2_name

        if save and mode != "l1" :
            if not isdir(self.savedir): makedirs(self.savedir)
            write_fits(self.savedir + "/maskR2" + self.name, self.mask.numpy())

    def estimate(self, w_r=0.03, w_r2=0.03, w_pcent=True, estimI="Both", med_sub=False, weighted_rot=True, res_pos=False,
                 w_way=(0, 1), maxiter=10, gtol=1e-10, kactiv=0, kdactiv=None, save="./", suffix = '', gif=False,
                 verbose=False, history=True, init_maxL=True, min_sub=True):
        """ Resole the minimization of probleme neo-mayo
            The first step with pca aim to find a good initialisation
            The second step process to the minimization
         
        Parameters
        ----------

        w_r : float
            Weight regularization, hyperparameter to control R1 regularization (smooth regul)

        w_r2 : float
            Weight regularization, hyperparameter to control R2 regularization (mask regul)

        w_pcent : bool
            Determine if regularization weight are raw value or percentage
            Pecentage a computed based on either the initaial values of L and X with PCA
            If kactive is set, compute the value based on the L and X at activation.

        estimI : str
            If "None" : normal minimization
            if "L" : With estimate a flux variation of the speakles map
            if "Frame" : With estimate a flux variation between each frame
            if "Both" : Will estimate both flux variation between each frame and bewteen frame's speakles map

        med_sub : bool
            If True, will proceed to a median subtraction (recommended)

        weighted_rot : bool
            if True, each frame will be weighted by the delta of rotation.
            see neo-mayo technical details to know more.

        res_pos : bool
            Add regularization to penalize negative residual

        w_way : tuple ints
            If (1,0) : ADI model constructed with the cube and rotate R (direct way)
            If (0,1) : ADI model constructed with the derotated cube and rotate L (reverse way)
            iI (1,1) : will do both

        maxiter : int
            Maximum number of optimization steps

        gtol : float
            Gradient tolerance; Set the break point of minimization.
            The break point is define as abs(J_n - J-(n-1)) < gtol

        kactiv : int or {"converg"}
            Activate regularization at a specific iteration
            If set to "converg", will be activated when break point is reach
            Equivalent to re-run a minimization with regul and L and X initialized at converged value

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
        self.savedir = ( save if isinstance(save, str) else "." ) + "/mustard_out"+self.name+"/"
        self.config[4] = estimI + "X and L" if estimI=="Both" else estimI
        overwrite = "\n" if history else "\r"
        ending = "undifined"

        # Keep track on history
        mink  = 2  # Min number of iter before convergence
        loss_evo = []; grad_evo = []

        # Regularization activation init setting
        if kactiv == "converg" : kactiv = maxiter  # If option converge, kactiv start when miniz end
        Ractiv = 0 if kactiv else 1  # Ractiv : regulazation is curently activated or not
        w_rp = w_r, w_r2 if w_pcent else  0, 0 # If values are in percent save them; It will be computed later
        w_r = w_r if w_r else 0; w_r2 = w_r2 if w_r2 else 0 # If None/0/False, it will be set to 0

        # Compute weights for small angles bias
        if weighted_rot :
            self.config[3] = "with"
            ang_weight = compute_rot_weight(self.model.rot_angles)
            self.ang_weight = torch.from_numpy(ang_weight.reshape((self.nb_frame, 1, 1, 1))).double()

        # ______________________________________
        # Define constantes and convert arry to tensor

        # Med sub
        if med_sub :
            meds = np.median(self.coro * self.science_data, (1, 2))
            for frame in range(self.nb_frame) :
                self.science_data[frame] = (self.science_data[frame] - meds[frame]).clip(0)

        if min_sub: self.decompose()

        science_data = torch.unsqueeze(torch.from_numpy(self.science_data), 1).double()
        science_data_derot_np = cube_derotate(self.science_data, self.model.rot_angles)
        science_data_derot = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 1).double()
        # med = torch.median(self.coro * science_data, dim=0, keepdim=True).values
        # med = np.median(abs(self.science_data))
        # med = 0

        # __________________________________
        # Initialisation with max common

        if self.L0x0 is not None:
            warnings.warn(DeprecationWarning("Use of initalization other than max-common is not recommended."))

        elif init_maxL :
            res = np.min(self.science_data, 0)
            R_fr = science_data_derot_np - cube_derotate(np.tile(res, (self.nb_frame, 1, 1)), -self.model.rot_angles)
            self.L0x0 = res.clip(min=0), np.mean(R_fr, axis=0).clip(min=0)
        else :
            res  = np.min(science_data_derot_np, 0)
            L_fr = self.science_data - cube_derotate(np.tile(res, (self.nb_frame, 1, 1)), self.model.rot_angles)
            self.L0x0 = np.mean(L_fr, axis=0), res

        L0, X0 = self.L0x0[0], self.L0x0[1]
        if self.config[0] == "peak_preservation" : self.xmax = np.max(X0)

        # __________________________________
        # Initialisation with max common

        L0 = torch.unsqueeze(torch.from_numpy(L0), 0).double()
        X0 = torch.unsqueeze(torch.from_numpy(X0), 0).double()
        flux_0 = torch.ones(self.model.nb_frame - 1)
        fluxR_0 = torch.ones(self.model.nb_frame - 1)

        # ______________________________________
        #  Init variables and optimizer

        Lk, Xk, flux_k, fluxR_k,  k = L0.clone(), X0.clone(), flux_0.clone(), fluxR_0.clone(), 0
        Lk.requires_grad = True; Xk.requires_grad = True

        # Model and loss at step 0
        with torch.no_grad():

            Y0 = self.model.forward(L0, X0, flux_0, fluxR_0) if w_way[0] else 0
            Y0_reverse = self.model.forward_ADI_reverse(L0, X0, flux_0, fluxR_0) if w_way[1] else 0

            loss0 = w_way[0] * torch.sum(self.ang_weight * self.coro * (Y0 - science_data) ** 2) + \
                    w_way[1] * torch.sum(self.ang_weight * self.coro * (Y0_reverse - science_data_derot) ** 2)

            if res_pos :
                Rx_smooth = self.model.get_Rx(X0, flux_0)
                Lf = self.model.get_Lf(L0, flux_0, fluxR_0)
                nospeck = ReLU(science_data - Lf)
                Rpos = torch.sum(self.ang_weight * self.coro * ReLU(Y0 - science_data) ** 2)
                self.map_negR0 = ReLU(Y0 - science_data)
                self.map_neg0 = Y0 - science_data

                # if w_way[0] and res_pos else 0
                # Rpos += torch.sum(self.ang_weight * self.coro * ReLU(Y0_reverse - science_data_derot-med) ** 2) \
                    # if w_way[1] and res_pos else 0
                Rpos *= 1#self.nb_frame-1 #self.nb_frame-1 #**2 # Rpos weight
            else : Rpos = 0

            if w_pcent and Ractiv : # Auto hyperparameters
                reg1 = self.regul(X0, L0)
                w_r  = w_rp[0] * loss0 / reg1  if w_rp[0] and reg1 > 0 else 0

                reg2 = self.regul2(X0, L0, self.mask)
                w_r2 = w_rp[1] * loss0 / reg2 if w_rp[1] and reg2 > 0 else 0

                if (w_rp[0] and reg1 < 0) or (w_rp[1] and reg2 < 0):
                    if verbose : print("Impossible to compute regularization weight. Activation is set to iteration n°2. ")
                    kactiv = 2

            R1_0 = Ractiv * w_r * self.regul(X0, L0) if w_r else 0
            R2_0 = Ractiv * w_r2 * self.regul2(X0, L0, self.mask) if w_r2 else 0
            loss0 += (R1_0 + R2_0 + Rpos)

        loss, R1, R2 = loss0, R1_0, R2_0

        # Starting minization soon ...
        stat_msg = init_msg.format(self.name, save, *self.config, w_r, w_r2, str(maxiter))
        if verbose: print(stat_msg)
        txt_msg = stat_msg

        # ____________________________________
        # Nested functions

        # Definition of minimizer step.
        def closure():
            nonlocal R1, R2, Rpos, loss, w_r, w_r2, Lk, Xk, flux_k, fluxR_k
            optimizer.zero_grad()  # Reset gradients

            # Compute model(s)
            Yk = self.model.forward(Lk, Xk, flux_k, fluxR_k) if w_way[0] else 0
            Yk_reverse = self.model.forward_ADI_reverse(Lk, Xk, flux_k, fluxR_k) if w_way[1] else 0

            # Compute regularization(s)
            R1 = Ractiv * w_r * self.regul(Xk, Lk)  if Ractiv * w_r else 0
            R2 = Ractiv * w_r2 * self.regul2(Xk, Lk, self.mask) if Ractiv * w_r2 else 0

            if res_pos :
                Rx_smooth = self.model.get_Rx(Xk, flux_k)
                Lf = self.model.get_Lf(Lk, flux_k, fluxR_k)
                nospeck = ReLU(science_data - Rx_smooth)
                Rpos = torch.sum(self.ang_weight * self.coro * ReLU(Yk - science_data) ** 2)
                self.map_neg = Yk - science_data
                self.map_negR = ReLU(Yk - science_data)
                Rpos *=  1 #self.nb_frame-1 #**2 #Rpos weight
            else : Rpos = 0

            # Compute loss and local gradients
            loss = w_way[0] * torch.sum( self.ang_weight * self.coro * (Yk - science_data) ** 2) + \
                   w_way[1] * torch.sum( self.ang_weight * self.coro * (Yk_reverse - science_data_derot) ** 2) + \
                   (R1 + R2 + Rpos)

            loss.backward()
            return loss

        # Definition of regularization activation
        def activation():
            nonlocal w_r, w_r2, optimizer, Xk, Lk, flux_k, fluxR_k
            for activ_step in ["ACTIVATED", "AJUSTED"]:  # Activation in two step

                # Second step : re-compute regul after performing a optimizer step
                if activ_step == "AJUSTED": optimizer.step(closure)

                with torch.no_grad():
                    if w_pcent:
                        reg1 = self.regul(Xk, Lk)
                        w_r  = w_rp[0] * (loss-Rpos) / reg1 if w_rp[0] and reg1 > 0 else 0

                        reg2 = self.regul2(Xk, Lk, self.mask)
                        w_r2 = w_rp[1] * (loss-Rpos) / reg2 if w_rp[1] and reg2 > 0 else 0

                    if (w_rp[0] and reg1 < 0) or (w_rp[1] and reg2 < 0):
                        if verbose: print("Impossible to compute regularization weight.")

                # Define the varaible to be estimated.
                if estimI == "Both":
                    optimizer = optim.LBFGS([Lk, Xk, flux_k, fluxR_k])
                    flux_k.requires_grad = True; fluxR_k.requires_grad = True
                elif estimI == "Frame":
                    optimizer = optim.LBFGS([Lk, Xk, flux_k])
                    flux_k.requires_grad = True
                elif estimI == "L":
                    optimizer = optim.LBFGS([Lk, Xk, fluxR_k])
                    fluxR_k.requires_grad = True
                elif estimI == "Justx":
                    optimizer = optim.LBFGS([Xk, fluxR_k])
                    Lk.requires_grad = False
                    fluxR_k.requires_grad = True
                elif estimI == "Justl":
                    optimizer = optim.LBFGS([Lk, fluxR_k])
                    Xk.requires_grad = False
                    fluxR_k.requires_grad = True
                else:
                    optimizer = optim.LBFGS([Lk, Xk])

                if activ_step == "AJUSTED":
                    process_to_prints(activ_msg.format(activ_step, w_r, w_r2), -0.5)

        # Definition of print routines
        def process_to_prints(extra_msg=None, sub_iter=0, last=False):
            nonlocal txt_msg
            iter_msg = info_iter.format(k + sub_iter, *loss_ratio(Ractiv, float(R1),float(R2), float(Rpos),
                                                                  float(loss)), loss)
            if extra_msg is not None : txt_msg += "\n" + extra_msg
            txt_msg += "\n" + iter_msg
            est_info = stat_msg.split("\n", 4)[3] + '\n'
            if gif: print_iter(Lk, Xk, flux_k, k + sub_iter, est_info + iter_msg, extra_msg, save, self.coro)
            if verbose: print(iter_msg, end=overwrite if not last else "\n\n")

        # Define the varaible to be estimated.
        if estimI == "Both":
            optimizer = optim.LBFGS([Lk, Xk, flux_k, fluxR_k])
            flux_k.requires_grad = True
            fluxR_k.requires_grad = True
        elif estimI == "Frame" :
            optimizer = optim.LBFGS([Lk, Xk, flux_k])
            flux_k.requires_grad = True
        elif estimI == "L":
            optimizer = optim.LBFGS([Lk, Xk, fluxR_k])
            fluxR_k.requires_grad = True
        elif estimI == "Justx":
            optimizer = optim.LBFGS([Xk, fluxR_k])
            Lk.requires_grad = False
            fluxR_k.requires_grad = True
        elif estimI == "Justl":
            optimizer = optim.LBFGS([Lk, fluxR_k])
            Xk.requires_grad = False
            fluxR_k.requires_grad = True
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

        try :
            for k in range(1, maxiter+1):

                # Activation
                if kactiv and k == kactiv:
                    Ractiv = 1; mink = k + 2
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
                if k == 1 : self.first_iter = (Lk, Xk, flux_k, fluxR_k) if estimI else (Lk, Xk)

                # Break point (based on gtol)
                grad = torch.mean(abs(Xk.grad.data)) if not estimI == "Justl" else torch.mean(abs(Lk.grad.data))
                grad_evo.append(grad)
                if k > mink and  (grad < gtol) : #or loss == loss_evo[-1] ):
                    if not Ractiv and kactiv:  # If regul haven't been activated yet, continue with regul
                        Ractiv = 1; mink = k+2; kactiv=k
                        activation()
                    else :
                        ending = 'max iter reached' if k == maxiter else 'Gtol reached'
                        break
                elif torch.isnan(loss) : # Also break if an error occure.
                    ending = 'Nan values end minimization. Last value estimated will be returned.'
                    break

                process_to_prints()

        except KeyboardInterrupt:
            ending = "keyboard interruption"

        process_to_prints(last=True)
        end_msg = "Done ("+ending+") - Running time : " + str(datetime.now() - start_time)
        if verbose: print(end_msg)
        txt_msg += "\n\n" + end_msg

        # ______________________________________
        # Done, store and unwrap results back to numpy array!

        if k > 1 and (torch.isnan(loss) or loss > loss_evo[-2]) and self.final_estim is not None:
            L_est = abs(self.final_estim[0].detach().numpy()[0])
            X_est = abs((self.coro * self.final_estim[1]).detach().numpy()[0])
        else :
            L_est, X_est = abs(Lk.detach().numpy()[0]), abs((self.coro * Xk).detach().numpy()[0])

        flux  = abs(flux_k.detach().numpy())
        fluxR = abs(fluxR_k.detach().numpy())
        loss_evo = [float(lossk.detach().numpy()) for lossk in loss_evo]

        # Result dict
        res = {'state'   : optimizer.state,
               'x'       : (L_est, X_est),
               'ambig'   : (L_est, X_est),
               'flux'    : flux,
               'fluxR'   : fluxR,
               'loss_evo': loss_evo,
               'Kactiv'  : kactiv,
               'ending'  : ending}

        self.res = res

        # Save
        if gif : iter_to_gif(save, self.name)

        if save :
            if not isdir(self.savedir): makedirs(self.savedir)
            write_fits(self.savedir + "/L_est"+self.name, L_est), write_fits(self.savedir + "/X_est"+self.name, X_est)
            with open(self.savedir + "/config.txt", 'w') as f:  f.write(txt_msg)
            if estimI :
                write_fits(self.savedir + "/flux"+self.name, flux)
                write_fits(self.savedir + "/fluxR"+self.name, fluxR)

        if estimI: return L_est, X_est, flux
        else     : return L_est, X_est


    def decompose(self):
        self.speckles = np.min(self.science_data, 0)
        self.ambiguities = np.min(cube_derotate(
            np.tile(self.speckles, (self.nb_frame, 1, 1)), self.model.rot_angles), 0)
        self.stellar_halo = np.min(cube_derotate(np.tile(self.ambiguities, (50, 1, 1)),
                                                 np.linspace(0, 360, 50)), 0)
        self.science_data_ori = self.science_data.copy()
        self.science_data = self.science_data - self.speckles + self.stellar_halo

    def get_science_data(self):
        """Return input cube and angles"""
        return self.model.rot_angles, self.science_data

    def get_residual(self, way = "direct", save=False):
        """Return input cube and angles"""

        if way == "direct" :
            science_data = torch.unsqueeze(torch.from_numpy(self.science_data), 1).double()
            reconstructed_cube = self.model.forward(*self.last_iter)  # Reconstruction on last iteration
            residual_cube = science_data - reconstructed_cube

        elif way == "reverse":
            science_data_derot_np = cube_derotate(self.science_data, self.model.rot_angles)
            science_data = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 1).double()
            reconstructed_cube = self.model.forward_ADI_reverse(*self.last_iter)  # Reconstruction on last iteration
            residual_cube = science_data - reconstructed_cube

        else : raise(ValueError,"way sould be 'reverse' or 'direct'")


        nice_residual = self.coro.detach().numpy() *  residual_cube.detach().numpy()[:, 0, :, :]
        if save:
            savedir = save  if isinstance(save,str) else self.savedir
            if not isdir(savedir): makedirs(savedir)
            write_fits(savedir + "/residual_"+way+"_"+ self.name, nice_residual)

        return nice_residual

    def get_evo_convergence(self, show=True, save=False):
        """Return loss evolution"""

        loss_evo = self.res['loss_evo']
        Kactiv = self.res["Kactiv"] + 1 if isinstance(self.res["Kactiv"], (int, float)) else len(loss_evo)-1

        if show : plt.ion()
        else : plt.ioff()

        fig = plt.figure("Evolution of loss criteria", figsize=(16, 9))
        fig.subplots(1, 2, gridspec_kw={'width_ratios': [Kactiv, len(loss_evo)-Kactiv]})

        plt.subplot(121), plt.xlabel("Iteration"), plt.ylabel("Loss - log scale"), plt.yscale('log')
        plt.plot(loss_evo[:Kactiv], 'X-', color="tab:orange"), plt.title("Loss evolution BEFORE activation")

        plt.subplot(122), plt.xlabel("Iteration"), plt.ylabel("Loss - log scale"), plt.yscale('log')
        plt.plot(loss_evo[Kactiv:], 'X-', color="tab:blue"), plt.title("Loss evolution, AFTER activation")
        plt.xticks(range(len(loss_evo)-Kactiv), range(Kactiv, len(loss_evo)) )

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
            write_fits(self.savedir + "/speckles_and_stellar_halo" + self.name, self.speckles+self.ambiguities)

        if show :
            plt.figure("Speckles map", figsize=(16, 9))
            plt.imshow(self.speckles, cmap='jet')
            plt.show()

        return self.speckles

    def get_ambiguity(self, show=True, save=False):

        if self.ambiguities is None :
            res = np.min(self.science_data, 0)
            self.ambiguities = np.min(cube_derotate(np.tile(res, (self.nb_frame, 1, 1)), -self.model.rot_angles), 0)

            self.stellar_halo = np.min(cube_derotate(np.tile(self.ambiguities, (50, 1, 1)),
                                                 np.linspace(0, 360, 50)), 0)

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

        if show : plt.ion()
        else : plt.ioff()

        plt.figure("Frame weight based on PA", figsize=(16, 9))

        plt.subplot(211), plt.xlabel("Frame"), plt.ylabel("PA in deg")
        plt.plot(rot_angles, 'X-', color="tab:purple"), plt.title("Angle for each frame")

        plt.subplot(212), plt.bar(range(len(weight)), weight,  color="tab:cyan", edgecolor="black")
        plt.title("Assigned weight"), plt.xlabel("Frame"), plt.ylabel("Frame weight")
        if show: plt.show()

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            plt.savefig(self.savedir + "/PA_frame_weight_" + self.name)

        return weight


    def mustard_results(self, per_vmax=99, r_no_scale=False):
        """Return loss evolution"""

        L, X = self.res["x"]
        cube = self.science_data
        ang  = self.model.rot_angles

        noise = self.get_residual()
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
            plt.imshow(cube[num], vmax=vmax, vmin=vmin, cmap='jet')
            plt.text(20, 40, "ADI cube", font)
            plt.title("Frame n°" + str(num))
            font["size"] = 22
            plt.text(20, 55, r'$\Delta$ Flux : 1{:+.2e}'.format(1 - flx[num]), font)

            font["size"] = 22
            plt.subplot(2, 2, 4)
            plt.imshow(flx[num] * frame_rotate(X, ang[num]), vmax=Rvmax, vmin=vmin, cmap='jet')
            plt.text(20, 40, "Rotate", font)

            font["size"] = 16
            plt.subplot(2, 4, 4)
            plt.imshow(flxR[num] * flx[num]* L, vmax=vmax, vmin=vmin, cmap='jet')
            plt.text(20, 40, "Static", font)
            font["size"] = 12
            plt.text(20, 55, r'$\Delta$ Flux : 1{:+.2e}'.format(1 - flxR[num]), font)

            font["size"] = 16
            font["color"] = "red"

            plt.subplot(2, 4, 3)
            plt.imshow(noise[num], cmap='jet')
            plt.clim(-np.percentile(noise[num], 98), +np.percentile(noise[num], 98))
            plt.text(20, 40, "Random", font)
            if show : plt.show()

        # ax_slid = plt.axes([0.1, 0.25, 0.0225, 0.63])
        # handler = Slider(ax=ax_slid, label="Frame", valmin=0, valmax=len(cube), valinit=0, orientation="vertical")
        # handler.on_changed(plot_framek)

        plt.ioff()
        plt.figure("TMP_MUSTARD", figsize=(16, 14))
        if not isdir(self.savedir): makedirs(self.savedir)
        if not isdir(self.savedir+"/tmp/"): makedirs(self.savedir+"/tmp/")

        images = []
        for num in range(len(cube)):
            plt.cla();plt.clf()
            plot_framek(num, show=False)
            plt.savefig(self.savedir+"/tmp/noise_" + str(num) + ".png")

        for num in range(len(cube)):
            images.append(Image.open(self.savedir+"/tmp/noise_" + str(num) + ".png"))

        for num in range(len(cube)):
            try: remove(self.savedir + "/tmp/noise_" + str(num) + ".png")
            except Exception as e: print("[WARNING] Failed to delete iter .png : " + str(e))

        try : rmdir(self.savedir+"/tmp/")
        except Exception as e : print("[WARNING] Failed to remove iter dir : " + str(e))

        images[0].save(fp=self.savedir+"MUSTARD.gif", format='GIF',
                       append_images=images, save_all=True, duration=200, loop=0)

        plt.close("TMP_MUSTARD")
        plt.ion()

    def mustard_extrem_decomposition(self, per_vmax=99, r_no_scale=False):
        """Return loss evolution"""

        L, X = self.res["x"]

        if self.science_data_ori is None: self.decompose()

        cube = self.science_data_ori
        cube_no_speck = self.science_data
        ang  = self.model.rot_angles

        noise = self.get_residual()
        flx, flxR = self.get_flux(show=False)
        flx = [1] + list(flx)
        flxR = [1] + list(flxR)

        font = {'color': 'white',
                'weight': 'bold',
                'size': 16,
                }

        vmax = np.percentile(cube, per_vmax)
        vmin = cube.min()

        vmax_nos = np.percentile(cube_no_speck, per_vmax)
        vmin_nos = cube_no_speck.min()

        Rvmax = np.percentile(X, 100) if r_no_scale else vmax

        def plot_speakles_and_halo(val: int, show=True) -> None:

            num = int(val)
            font["size"] = 22
            font["color"] = "white"

            plt.subplot(3, 3, 1)
            plt.imshow(cube[num], vmax=vmax, vmin=vmin, cmap='jet')
            plt.text(20, 40, "ADI cube", font)
            plt.title("Frame n°" + str(num))

            plt.subplot(3, 3, 2)
            plt.imshow(cube_no_speck[num], vmax=vmax_nos, vmin=vmin_nos, cmap='jet')
            plt.text(20, 40, "ADI cube without Speckles", font)
            plt.title("Frame n°" + str(num))

            plt.subplot(3, 3, 3)
            plt.imshow(cube_no_speck[num] - self.stellar_halo, cmap='jet')
            plt.text(20, 40, "ADI cube without Speckles and stellar halo", font)
            plt.title("Frame n°" + str(num))

            plt.subplot(3, 2, 3)
            plt.imshow(self.speckles, vmax=vmax, vmin=vmin, cmap='jet')
            plt.text(20, 40, "Speckles without ambiguities", font)

            plt.subplot(3, 2, 4)
            plt.imshow(self.speckles + self.ambiguities - self.stellar_halo, vmax=vmax, vmin=vmin, cmap='jet')
            plt.text(20, 40, "Speckles + unclassified signal", font)

            plt.subplot(3, 3, 7)
            plt.imshow(flx[num] * frame_rotate(self.ambiguities, ang[num]), vmax=vmax_nos, vmin=vmin_nos, cmap='jet')
            plt.text(20, 40, "Ambiguities", font)

            plt.subplot(3, 3, 8)
            plt.imshow(self.stellar_halo, vmax=vmax_nos, vmin=vmin_nos, cmap='jet')
            plt.text(20, 40, "Stellar halo", font)

            plt.subplot(3, 3, 9)
            plt.imshow(flx[num] * frame_rotate(self.ambiguities - self.stellar_halo, ang[num]), vmax=vmax_nos, vmin=vmin_nos, cmap='jet')
            plt.text(20, 40, "Unclassified", font)

            if show : plt.show()

        # ax_slid = plt.axes([0.1, 0.25, 0.0225, 0.63])
        # handler = Slider(ax=ax_slid, label="Frame", valmin=0, valmax=len(cube), valinit=0, orientation="vertical")
        # handler.on_changed(plot_framek)

        plt.ioff()
        plt.figure("TMP_MUSTARD", figsize=(16, 14))
        if not isdir(self.savedir): makedirs(self.savedir)
        if not isdir(self.savedir+"/tmp/"): makedirs(self.savedir+"/tmp/")

        images = []
        for num in range(len(cube_no_speck)):
            plt.cla();plt.clf()
            plot_speakles_and_halo(num, show=False)
            plt.savefig(self.savedir+"/tmp/noise_" + str(num) + ".png")

        for num in range(len(cube_no_speck)):
            images.append(Image.open(self.savedir+"/tmp/noise_" + str(num) + ".png"))

        for num in range(len(cube_no_speck)):
            try: remove(self.savedir + "/tmp/noise_" + str(num) + ".png")
            except Exception as e: print("[WARNING] Failed to delete iter .png : " + str(e))

        try : rmdir(self.savedir+"/tmp/")
        except Exception as e : print("[WARNING] Failed to remove iter dir : " + str(e))

        images[0].save(fp=self.savedir+"DECOMPO.gif", format='GIF',
                       append_images=images, save_all=True, duration=200, loop=0)

        plt.close("TMP_MUSTARD")
        plt.ion()


    def get_flux(self, show=True, save=False):
        """Return relative flux variations between frame"""

        flux = self.res['flux']
        fluxR = self.res['fluxR']

        if show:
            plt.ion()
        else:
            plt.ioff()

        plt.figure("Relative flux variations between frame", figsize=(16, 9))
        lim = max(abs((flux-1)))
        limR = max(abs((fluxR-1)))
        if lim==0  : lim+= 1
        if limR==0 : limR+= 1

        plt.subplot(1, 2, 1), plt.bar(range(len(flux)), flux-1, bottom=1, color='tab:red', edgecolor="black")
        plt.ylabel("Flux variation"), plt.xlabel("Frame"), plt.title("Flux variations between Frames")
        plt.ylim([1-lim, 1+lim]), plt.ticklabel_format(useOffset=False)

        plt.subplot(1, 2, 2), plt.bar(range(len(fluxR)), fluxR-1, bottom=1, color='tab:green', edgecolor="black")
        plt.ylabel("Flux variation"), plt.xlabel("Frame"), plt.title("Flux variations of starlight map")
        plt.ylim([1-limR, 1+limR]), plt.ticklabel_format(useOffset=False)
        if show: plt.show()

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            plt.savefig(self.savedir + "/flux_" + self.name)

        return flux, fluxR

    def get_cube_without_speckles(self, way="direct", save=False):
        """Return input cube and angles"""

        Lk, _ , flux_k, fluxR_k = self.last_iter
        if way == "direct":
            science_data = torch.unsqueeze(torch.from_numpy(self.science_data), 1).double()
            reconstructed_cube = science_data - self.model.get_Lf(Lk, flux_k, fluxR_k)

        elif way == "reverse":
            science_data_derot_np = cube_derotate(self.science_data, self.model.rot_angles)
            science_data = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 1).double()
            reconstructed_cube = science_data - self.model.get_Rx(Lk, flux_k, fluxR_k, inverse=True)

        else : raise(ValueError,"way sould be 'reverse' or 'direct'")

        reconstructed_cube = reconstructed_cube.detach().numpy()[:, 0, :, :]
        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            write_fits(self.savedir + "/cube_without_speckles_" + way + "_" + self.name, reconstructed_cube)

        return reconstructed_cube


    def get_reconstruction(self, way="direct", save=False):
        """Return input cube and angles"""

        if way == "direct":
            science_data = torch.unsqueeze(torch.from_numpy(self.science_data), 1).double()
            reconstructed_cube = self.model.forward(*self.last_iter)  # Reconstruction on last iteration

        elif way == "reverse":
            science_data_derot_np = cube_derotate(self.science_data, self.model.rot_angles)
            science_data = torch.unsqueeze(torch.from_numpy(science_data_derot_np), 1).double()
            reconstructed_cube = self.model.forward_ADI_reverse(*self.last_iter)  # Reconstruction on last iteration

        else : raise(ValueError,"way sould be 'reverse' or 'direct'")

        reconstructed_cube = self.coro.numpy() * reconstructed_cube.detach().numpy()[:, 0, :, :]
        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            write_fits(self.savedir + "/reconstruction_" + way + "_" + self.name, reconstructed_cube)

        return reconstructed_cube

    def get_initialisation(self, save=False):
        """Return input cube and angles"""

        if self.L0x0 is None : raise "No initialisation have been performed"
        L0, X0 = self.L0x0

        if save:
            if not isdir(self.savedir): makedirs(self.savedir)
            if not isdir(self.savedir+"/L0X0/"): makedirs(self.savedir+"/L0X0/")
            print("Save init from in " + self.savedir+"/L0X0" + "...")

            nice_X0 = self.coro.numpy() * X0

            write_fits(self.savedir+"/L0X0/" + "/L0.fits", L0, verbose=False)
            write_fits(self.savedir+"/L0X0/" + "/X0.fits", X0, verbose=False)
            write_fits(self.savedir+"/L0X0/" + "/nice_X0.fits", self.coro.numpy() * nice_X0, verbose=False)

        return L0, X0

    def set_savedir(self, savedir: str):
        self.savedir = savedir

# %% ------------------------------------------------------------------------
def normlizangle(angles: np.array) -> np.array:
    """Normaliz an angle between 0 and 360"""

    angles[angles < 0] += 360
    angles = angles % 360

    return angles


def compute_rot_weight(angs: np.array) -> np.array:
    nb_frm = len(angs)

    # Set value of the delta offset max
    max_rot = np.median(abs(angs[:-1]-angs[1:]))
    if max_rot > 1 : max_rot = 1

    ang_weight = []
    for id_ang, ang in enumerate(angs):
        # If a frame neighbours delta-rot under max_rot, frames are less valuable than other frame (weight<1)
        # If the max_rot is exceed, the frame is not consider not more valuable than other frame (weight=1)
        # We do this for both direction's neighbourhoods

        nb_neighbours = 0; detla_ang = 0
        tmp_id = id_ang

        while detla_ang < max_rot and tmp_id < nb_frm - 1:  # neighbours after
            tmp_id += 1; nb_neighbours += 1
            detla_ang += abs(angs[tmp_id] - angs[id_ang])
        w_1 = (1 + abs(angs[id_ang] - angs[id_ang + 1])) / nb_neighbours if nb_neighbours > 1 else 1

        tmp_id = id_ang; detla_ang = 0
        nb_neighbours = 0
        while detla_ang < max_rot and tmp_id > 0:  # neighbours before
            tmp_id -= 1; nb_neighbours += 1
            detla_ang += abs(angs[tmp_id] - angs[id_ang])
        w_2 = (1 + abs(angs[id_ang] - angs[id_ang - 1])) / nb_neighbours if nb_neighbours > 1 else 1

        ang_weight.append((w_2 + w_1) / 2)

    return np.array(ang_weight)
