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

from neo_mayo.utils import unpack_science_datadir
from vip_hci.fits import open_fits, write_fits
from os import mkdir
from os.path import isdir

# Algo and science model
from neo_mayo.algo import init_estimate, sobel_tensor_conv
import torch
import numpy as np

# Loss functions
from torch.nn import MSELoss
import torch.optim as optim
from torch import sum as tsum

# Other
from neo_mayo.utils import circle, iter_to_gif, print_iter
from neo_mayo.model import model_ADI

# -- For verbose -- #
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
sep = ('_' * 50)

info_iter = "Iteration n°{} : total loss {:.6e}" +\
            "\n\tWith R1 = {:.4e} ({:.0f}%)" +\
            "\n\tWith R2 = {:.4e} ({:.0f}%)" +\
            "\n\tand loss = {:.4e} ({:.0f}%)"

init_msg = sep + "\nResolving mayo optimization problem ..." +\
                 "\nMinimiz LBFGS with '{}' loss and '{}' regul" +\
                 "\nw_r are sets to w_r={:.2e} and w_r2={:.2e}, maxiter={}\n"

loss_ratio = lambda Ractiv, R1, R2, L: tuple(np.array((1, 100/L)) * R1) + \
                               tuple(np.array((1, 100/L)) * R2) + \
                               tuple(np.array((1, 100/L)) * (L - Ractiv * (R1 + R2)))


# %%

class mayo_estimator:
    """ Neo-mayo Algorithm main class 
   
    /!\ Neo-mayo isn't exaclty the same as Mayo pipeline. 
    Some choices have been re-thinked

    """

    def __init__(self, datadir="./data", coro=6, pupil="edge", rot="fft", loss="mse",
                 regul="smooth", Badframes=None, epsi=1e-3):

        # -- Create model and define constants --
        angles, science_data = unpack_science_datadir(datadir)
        if Badframes is not None:
            science_data = np.delete(science_data, Badframes, 0)
            angles = np.delete(angles, Badframes, 0)

        # Constants
        self.shape = science_data[0].shape
        self.nb_frames = science_data.shape[0]
        self.L0x0 = None
        self.mask = 1  # R2 mask default. Will be set if call R2 config

        # Coro and pupil masks
        if pupil is None: pupil = np.ones(self.shape); pupilR = pupil
        elif pupil == "edge":
            pupil  = circle(self.shape, self.shape[0]/2)
            pupilR = circle(self.shape, self.shape[0]/2 - 2)
        elif isinstance(pupil, float) or isinstance(pupil, int):
            pupil  = circle(self.shape, pupil)
            pupilR = circle(self.shape, pupil - 2)
        else: raise ValueError("Invalid pupil key argument. Possible values : {float/int, None, 'edge'}")

        if coro is None: coro = 0
        self.coro   = (1 - circle(self.shape, coro)) * pupil
        self.coroR  = (1 - circle(self.shape, coro+2)) * pupilR

        # Convert to tensor
        self.science_data = self.coro * science_data
        self.model = model_ADI(angles, self.coro, rot=rot)

        self.coro  = torch.from_numpy(self.coro).double()
        self.coroR = torch.from_numpy(self.coroR).double()

        self.science_data = torch.unsqueeze(torch.from_numpy(self.science_data), 1).double()

        # -- Define our minimization problem
        self.fun_loss = MSELoss(reduction='sum')

        if regul == "smooth_with_edges":
            self.regul = lambda X: torch.sum(self.coroR * sobel_tensor_conv(X) ** 2 - epsi ** 2)
        elif regul == "smooth":
            self.regul = lambda X: torch.sum(self.coroR * sobel_tensor_conv(X, axis='y') ** 2) +\
                                   torch.sum(self.coroR * sobel_tensor_conv(X, axis='x') ** 2)
        elif regul == "l1":
            self.regul = lambda X: torch.sum(self.coroR * torch.abs(X))

        self.regul2 = lambda X, L, M:  0

        self.config = regul, loss

    def initialisation(self, from_dir=None, save=None, **kwargs):
        """ The first step with greed aim to find a good initialisation 
         
         Parameters
         ----------
         save
         from_dir : str (optional)
             if None, compute iterative PCA
             if a path is given, get L0 (starlight) and X0 (circonstellar) from the directory
         
         Returns
         -------
         L_est, X_est: ndarray
             Estimated starlight (L) 
             and circunstlellar contributions (X) 
         
        """

        if from_dir:
            print("Get init from pre-processed datas...")
            L0 = open_fits(from_dir + "/L0.fits", verbose=False)
            X0 = open_fits(from_dir + "/X0.fits", verbose=False)

        else:

            # -- Iterative PCA for variable init -- #
            start_time = datetime.now()
            print(sep + "\nInitialisation  ...")

            L0, X0 = init_estimate(self.science_data, self.model.rot_angles, **kwargs)

            print("Done - running time : " + str(datetime.now() - start_time) + "\n" + sep)

            if save:
                if not isdir(save): mkdir(save)
                print("Save init from in " + save + "...")
                write_fits(save + "/L0.fits", L0, verbose=False)
                write_fits(save + "/X0.fits", X0, verbose=False)

        # -- Define constantes
        self.L0x0 = (L0, X0)

        return L0, X0

    def configR2(self, Msk=None, mode = "mask", penaliz = "X", invert=False):
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
        With mode dist :
            if "X" : R = dist(M-X)
            if "L" : L = -dist(M-L)
        With mode l1 :
            if "X" : R = sum(X)
            if "L" : L = -sum(L)

        invert : Bool
        Reverse penalize mode bewteen L and X.

        Returns
        -------

        """
        if mode != 'l1' :
            if isinstance(Msk, np.ndarray): Msk = torch.from_numpy(Msk)
            if not (isinstance(Msk, torch.Tensor) and Msk.shape == self.model.frame_shape):
                raise TypeError("Mask M should be tensor or arr y of size " + str(self.model.frame_shape))

        penaliz = penaliz.capitalize()
        rM = self.coroR  # corono mask for regul
        if penaliz not in ("X", "L", "Both", "B") :
            raise Exception("Unknown value of penaliz. Possible values are {'X','L','B'}")

        if mode == "dist":
            self.mask = Msk
            sign = -1 if invert else 1
            if   penaliz == "X"   :  self.regul2 = lambda X, L, M: sign * tsum( rM * (M - X) ** 2)
            elif penaliz == "L"   :  self.regul2 = lambda X, L, M: sign * tsum( rM * (M - X) ** 2)
            elif penaliz in ("Both", "B"):  self.regul2 = lambda X, L, M: sign * tsum( rM * (M - X) ** 2) -\
                                                                   sign * -tsum( rM * (M - L) ** 2)

        elif mode == "mask":
            Msk = Msk/torch.max(Msk)  # Normalize mask
            self.mask = (1-Msk) if invert else Msk
            if   penaliz == "X"   : self.regul2 = lambda X, L, M: tsum( rM * (M * X) ** 2)
            elif penaliz == "L"   : self.regul2 = lambda X, L, M: tsum( rM * ((1 - M) * L) ** 2)
            elif penaliz in ("Both", "B"): self.regul2 = lambda X, L, M: tsum( rM * (M * X) ** 2) + \
                                                                  tsum(M)**2/tsum((1 - M))**2 *\
                                                                  tsum( rM * ((1 - M) * L) ** 2)

        elif mode == "l1":
            if   penaliz == "X"   : self.regul2 = lambda X, L, M: tsum(X ** 2)
            elif penaliz == "L"   : self.regul2 = lambda X, L, M: -tsum(L ** 2)
            elif penaliz in ("Both", "B"): self.regul2 = lambda X, L, M: tsum(X ** 2)-tsum(L ** 2)

        else: raise Exception("Unknown value of mode. Possible values are {'mask','dist','l1'}")

    def estimate(self, w_r=0.03, w_r2=0.03, w_pcent=True, estimI=False, maxiter=10,
                 gtol=1e-10, kactiv=0, kdactiv=None, save=True, suffix = "", gif=False, verbose=False):
        """ Resole the minimization problem as describe in mayo
            The first step with greed aim to find a good initialisation 
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

        estimI : bool
            If True, estimate flux variation between each frame of data-science cube I_ks.
            ADI model will be written as I_k*L + R(X) = Y_k

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

        save : str or None
            if path is given, save the result as fits

        suffix : string
            String suffix to named the simulation outputs

        gif : bool
            if path is given, save each the minimization step as a gif

        verbose : bool
            Activate or deactivate print

         Returns
         -------
         L_est, X_est: ndarray
             Estimated starlight (L) and circunstlellar (X) contributions
         
        """

        if self.L0x0 is None: raise AssertionError("No L0/x0. You need to run initialisation")
        science_data = self.science_data
        mink  = 2  # Min number of iter before convergence
        loss_evo = []

        # Regularization activation init setting
        if kactiv == "converg" : kactiv = maxiter
        Ractiv = 0 if kactiv else 1
        if not kdactiv: kdactiv = maxiter

        # ______________________________________
        # Define constantes and convert arry to tensor

        science_data.requires_grad = False

        L0, X0 = self.L0x0

        L0 = self.coro * torch.unsqueeze(torch.from_numpy(L0), 0).double()
        X0 = self.coro * torch.unsqueeze(torch.from_numpy(X0), 0).double()
        flux_0 = torch.ones(self.model.nb_frame - 1)

        # ______________________________________
        #  Init variables and optimizer

        Lk, Xk, flux_k = L0, X0, flux_0
        Lk.requires_grad = True; Xk.requires_grad = True

        # Model and loss at step 0
        with torch.no_grad():

            Y0 = self.model.forward_ADI(L0, X0, flux_0)
            loss0 = self.fun_loss(Y0, science_data)

            if w_pcent and Ractiv :
                w_r  = w_r * loss0 / self.regul(X0)   # Auto hyperparameters
                w_r2 = w_r2 * loss0 / self.regul2(X0, L0, self.mask)

            R1_0 = Ractiv * w_r * self.regul(X0)
            R2_0 = Ractiv * w_r2 * self.regul2(X0, L0, self.mask)
            loss0 += (R1_0 + R2_0)

        loss, R1, R2 = loss0, R1_0, R2_0

        # Definition of minimizer step.
        def closure():
            nonlocal R1, R2, loss
            optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                Yk = self.model.forward_ADI(Lk, Xk, flux_k)
                R1 = Ractiv * w_r * self.regul(Xk)
                R2 = Ractiv * w_r2 * self.regul2(Xk, Lk, self.mask)
                loss = self.fun_loss(Yk, science_data) + (R1 + R2)
                loss.backward()
            return loss

        # If L_I is set to True, L variation intensity will be estimated
        if estimI :
            optimizer = optim.LBFGS([Lk, Xk, flux_k])
            flux_k.requires_grad = True
        else:
            optimizer = optim.LBFGS([Lk, Xk])

        # ______________________________________
        #  Minimizations Start

        print(init_msg.format(str(self.config[1]), str(self.config[0]), w_r, w_r2, str(maxiter)))
        start_time = datetime.now()

        # -- Minimization Start ! -- #
        for k in range(maxiter):

            # Save
            loss_evo.append(loss)

            # Prints
            if verbose: print(info_iter.format(k, loss, *loss_ratio(Ractiv, float(R1), float(R2), float(loss))))
            if gif: print_iter(Lk, Xk, k, loss, R1, R2, self.config, w_r, w_r2, Ractiv, estimI, flux_k, save)

            # Activation
            if kactiv and k == kactiv:  # Only activate regul after few iterations
                Ractiv = 1; mink = k+2
                with torch.no_grad():
                    if w_pcent:
                        w_r  = w_r * loss / self.regul(Xk)
                        w_r2 = w_r2 * loss / self.regul2(Xk, Lk, self.mask)
                    if estimI:
                        optimizer = optim.LBFGS([Lk, Xk, flux_k])
                        flux_k.requires_grad = True
                    else:
                        optimizer = optim.LBFGS([Lk, Xk])
                    print("REGUL HAVE BEEN ACTIVATED with w_r={:.2e} and w_r2={:.2e}".format(w_r, w_r2))

            if kdactiv and k == kdactiv: Ractiv = 0  # Shut down regul after few iterations

            # -- MINIMIZER STEP -- #
            optimizer.step(closure)

            # Break point
            if k > mink and (abs(loss_evo[-2] - loss_evo[-1])) < gtol:

                if not Ractiv and kactiv:
                    Ractiv = 1; mink = k+2
                    with torch.no_grad():
                        if w_pcent:
                            w_r  = w_r * loss / self.regul(Xk)
                            w_r2 = w_r2 * loss / self.regul2(Xk, Lk, self.mask)
                        if estimI:
                            optimizer = optim.LBFGS([Lk, Xk, flux_k])
                            flux_k.requires_grad = True
                        else:
                            optimizer = optim.LBFGS([Lk, Xk])
                    print("Convergence reached : REGUL HAVE BEEN ACTIVATED "
                          "\nwith w_r={:.2e} and w_r2={:.2e}".format(w_r, w_r2))

                else : break

        print("Done - Running time : " + str(datetime.now() - start_time))

        # ______________________________________
        # Done, store and unwrap results back to numpy array!

        L_est, X_est = abs(Lk.detach().numpy()[0, :, :]), abs(Xk.detach().numpy()[0, :, :])
        flux  = abs(flux_k.detach().numpy())

        # Result dict
        res = {'state'   : optimizer.state,
               'x'       : (L_est, X_est),
               'flux'     : flux,
               'loss_evo': loss_evo}

        self.res = res

        # Save
        if gif : iter_to_gif(save, suffix)

        suffix = '' if not suffix else '_' + suffix
        if save: write_fits(save + "/L_est"+suffix, L_est), write_fits(save + "/X_est"+suffix, X_est)
        if save and estimI : write_fits(save + "/flux"+suffix, flux)

        if estimI: return L_est, X_est, flux
        else     : return L_est, X_est

    def get_science_data(self):
        return self.model.rot_angles, self.science_data.numpy()[:, 0, :, :]
