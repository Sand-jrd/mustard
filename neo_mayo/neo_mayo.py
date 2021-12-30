#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:30:10 2021

______________________________

|      Neo-Mayo estimator     |
______________________________


@author: sand-jrd
"""

# For file managment

from neo_mayo.utils import unpack_science_datadir
from vip_hci.preproc import frame_rotate
from vip_hci.fits import open_fits, write_fits
from os import mkdir
from os.path import isdir

# Algos and science model
from neo_mayo.algo import init_estimate, sobel_tensor_conv
import torch
import numpy as np

# Loss functions
from torch.nn import MSELoss
import torch.optim as optim
from torch import sum as tsum
from torch import sqrt as tsqrt

# Other
from neo_mayo.utils import circle, iter_to_gif, print_iter
from neo_mayo.model import model_ADI

# For verbose
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
sep = ('_' * 50)


# %%


class mayo_estimator:
    """ Neo-mayo Algorithm main class 
   
    /!\ Neo-mayo isn't exaclty the same as Mayo pipeline. 
    Some choices have been re-thinked

    """

    def __init__(self, datadir="./data", mask_size=15, rot="fft", loss="mse",
                 regul="smooth", Badframes=None, epsi=1e-3):

        # -- Create model and define constants --
        angles, science_data = unpack_science_datadir(datadir)
        if Badframes is not None:
            science_data = np.delete(science_data, Badframes, 0)
            angles = np.delete(angles, Badframes, 0)

        self.shape = science_data[0].shape
        self.nb_frames = science_data.shape[0]
        self.L0x0 = None

        mask = circle(self.shape, mask_size)

        self.const = {"science_data": science_data}
        self.model = model_ADI(angles, mask, rot=rot)


        # -- Define our minimization problem
        self.fun_loss = MSELoss(reduction='sum')

        if regul == "smooth_with_edges":
            self.regul = lambda X: torch.sum(sobel_tensor_conv(X) ** 2 - epsi ** 2)
        elif regul == "smooth":
            self.regul = lambda X: torch.sum(sobel_tensor_conv(X) ** 2)
        elif regul == "l1":
            self.regul = lambda X: torch.sum(torch.abs(X))

        self.regul2 =  lambda X,L,M:  0

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

            L0, X0 = init_estimate(self.const["science_data"], self.model.rot_angles, **kwargs)

            print("Done - running time : " + str(datetime.now() - start_time) + "\n" + sep)

            if save:
                if not isdir(save): mkdir(save)
                print("Save init from in " + save + "...")
                write_fits(save + "/L0.fits", L0, verbose=False)
                write_fits(save + "/X0.fits", X0, verbose=False)

        # -- Define constantes
        self.L0x0 = (L0, X0)

        return L0, X0

    def configR2(self, M, mode = "mask", penaliz = "X", invert=False ):
        """ Configuration for Second regularization.
        Two possible mode :   R = M*X)      (mode mask)
                           or R = dist(M-X) (mode dist)

        Parameters
        ----------
        M : numpy.array or torch.tensor
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
        If "both" : will do both

        invert : Bool
        Reverse penalize mode bewteen L and X.

        Returns
        -------

        """
        if isinstance(M, np.ndarray) : M = torch.from_numpy(M)
        if not (isinstance(M, torch.Tensor) and M.shape==self.model.frame_shape) :
            raise TypeError("Mask M should be tensor or arr y of size " + str(self.model.frame_shape))

        N = self.model.frame_shape[0] ** 2

        if mode == "dist":
            self.mask = M
            sign = -1/N if invert else 1/N
            if   penaliz == "X"   :  self.regul2 = lambda X, L, M: sign * tsqrt(tsum((M - X) ** 2))
            elif penaliz == "L"   :  self.regul2 = lambda X, L, M: sign * tsqrt(tsum((M - X) ** 2))
            elif penaliz == "both":  self.regul2 = lambda X, L, M: sign * tsqrt(tsum((M - X) ** 2)) - sign * tsqrt(tsum((M - L) ** 2))
            else: raise Exception("Unknown value of penaliz. Possible values are 'X','L' or 'both'")

        elif mode == "mask":
            M = M/torch.max(M) # Normalize mask
            self.mask = (1-M)/N if invert else M/N
            if   penaliz == "X"   : self.regul2 = lambda X, L, M: tsqrt(tsum((M * X) ** 2))
            elif penaliz == "L"   : self.regul2 = lambda X, L, M: tsqrt(tsum(((1 - M) * L) ** 2))
            elif penaliz == "both": self.regul2 = lambda X, L, M: tsqrt(tsum((M * X) ** 2)) + tsqrt(tsum(((1 - M) * L) ** 2)) // 2
            else: raise Exception("Unknown value of penaliz. Possible values are 'X','L' or 'both'")

        else : raise Exception("Unknown value of mode. Possible values are 'mask' or 'dist'")

    def estimate(self, w_r=0, w_r2=0, maxiter=10, kactiv=0, kdactiv=None, estimI=False, save=False, suffix = "", gif=False, verbose=False):
        """ Resole the minimization problem as describe in mayo
            The first step with greed aim to find a good initialisation 
            The second step process to the minimization
         
        Parameters
        ----------

        w_r : float
            Weight regularization, hyperparameter to control regularization

        maxiter : int
            Maximum number of optimization steps

        save : str or None
            if path is given, save the result as fits
            
        gif : str or None
            if path is given, save each the minizations step as a gif

        verbose : bool
            Activate or desactivate print

         Returns
         -------
         L_est, X_est: ndarray
             Estimated starlight (L) and circunstlellar (X) contributions
         
        """

        if self.L0x0 is None: raise AssertionError("No L0/x0. You need to run initialisation")
        const = self.const
        Ractiv = 0 if kactiv else 1
        if not kdactiv: kdactiv = maxiter
        loss_evo = []

        # ______________________________________
        # Define constantes and convert arry to tensor

        science_data = torch.unsqueeze(torch.from_numpy(const["science_data"]), 1).double()
        science_data.requires_grad = False

        L0, X0 = self.L0x0

        L0 = torch.unsqueeze(torch.from_numpy(L0), 0).double()
        X0 = torch.unsqueeze(torch.from_numpy(X0), 0).double()

        # ______________________________________
        #  Init variables and optimizer

        Lk, Xk = L0, X0
        Lk.requires_grad = True; Xk.requires_grad = True
        flux_k = torch.ones(self.model.nb_frame - 1)

        # Model and loss at step 0
        Y = self.model.forward_ADI(Lk, Xk, flux_k)
        loss = self.fun_loss(Y, science_data)
        R1 = w_r * self.regul(Xk)
        R2 = w_r2 * self.regul2(Xk, Lk, self.model.mask)

        # Definition of minimizer step.
        def closure():
            nonlocal R1,R2, loss
            optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                Yk = self.model.forward_ADI(Lk, Xk, flux_k)
                R1 = w_r * self.regul(Xk) if w_r else 0
                R2 = w_r2 * self.regul2(Xk,Lk,self.mask) if w_r2 else 0
                loss = self.fun_loss(Yk, science_data) + Ractiv * (R1 + R2)
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

        start_time = datetime.now()
        print(sep + "\nResolving mayo optimization problem ..." +
              "\nMinimiz LBFGS with '" + str(self.config[1]) + "' loss and '" + str(self.config[0]) + "' regul\n" +
              "\nw_r are sets to w_r={:.2f} and w_r2={:.2f}".format(w_r,w_r2) + ", maxiter=" + str(maxiter) + "\n")

        for k in range(maxiter):
            loss_evo.append(loss)
            if verbose: print("Iteration n°" + str(k) + " {:.6e}".format(loss) +
                              "\n\tWith R1 = {:.4e} ({:.0f}%)".format(R1, 100 * Ractiv * R1 / loss) +
                              "\n\tWith R2 = {:.4e} ({:.0f}%)".format(R2, 100 * Ractiv * R2 / loss) +
                              "\n\tand loss = {:.4e} ({:.0f}%) \n".format(loss - Ractiv * (R1 + R2), 100 * (loss - Ractiv * (R1 + R2)) / loss))
            if gif: print_iter(Lk, Xk, k, loss, R1, R2, self.config, w_r, w_r2, Ractiv, estimI, flux_k, save)

            if kactiv and k > kactiv: Ractiv = 1  # Only activate regul after few iterations
            if kdactiv and k > kdactiv: Ractiv = 0  # Shut down regul after few iterations

            optimizer.step(closure)

        print("Done - Running time : " + str(datetime.now() - start_time))

        # ______________________________________
        # Done, Store and unwrap results back to numpy array!

        L_est, X_est = abs(Lk.detach().numpy()[0, :, :]), abs(Xk.detach().numpy()[0, :, :])
        flux  = abs(flux_k.detach().numpy())

        # Result dict
        res = {'state'   : optimizer.state,
               'x'       : (L_est, X_est),
               'flux'     : flux,
               'loss_evo': loss_evo}

        self.res = res

        # Save
        if gif: iter_to_gif(save, suffix)
        if save: write_fits(save + "/L_est"+suffix, L_est), write_fits(save + "/X_est"+suffix, X_est)
        if save and estimI : write_fits(save + "/flux"+suffix, flux)

        if estimI : return L_est, X_est, flux
        else   : return L_est, X_est

    # _____________________________________________________________
    # _____________ Tools functions of mayo_estimator _____________

    def remove_speckels(self, derot=False):
        """ Remove the speckle map estimation
        This can be better than the X estimation from the minimisation due to rotation flaws
        """
        if not hasattr(self, 'res'): raise AssertionError("Estimation should be lunched first")

        science_data = self.const["science_data"]
        L_est, X_est = self.res['x']
        Xs_est = science_data - L_est

        if derot:
            for frame in range(self.model.nb_frame): Xs_est[frame] = frame_rotate(Xs_est[frame])

        return Xs_est

    def get_science_data(self):
        return self.model.rot_angles, self.const["science_data"]
