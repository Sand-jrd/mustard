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
import matplotlib.pyplot as plt

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
from torch.nn import HuberLoss, MSELoss
import torch.optim as optim

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

    def __init__(self, datadir="./data", delta=1, mask_size=None, rot="torchvision", loss="mse", epsi=1e-3,
                 regul="smooth"):

        # -- Create model and define constants --
        angles, psf, science_data = unpack_science_datadir(datadir)

        if not mask_size: mask_size = psf.shape[0] // 2 - 2
        mask = circle(psf.shape, mask_size)

        self.const = {"science_data": science_data}
        self.model = model_ADI(angles, psf, mask, rot=rot)

        self.shape = psf.shape
        self.nb_frames = science_data.shape[0]
        self.L0x0 = None

        # -- Define our minimization problem
        if loss == 'huber':
            self.fun_loss = HuberLoss(reduction='sum', delta=delta)
        elif loss == 'mse':
            self.fun_loss = MSELoss(reduction='sum')

        if regul == "smooth_with_edges":
            self.regul = lambda X: torch.sum(torch.sqrt(torch.abs(sobel_tensor_conv(X) - epsi)))
        elif regul == "smooth":
            self.regul = lambda X: torch.sum(sobel_tensor_conv(X)**2)
        elif regul == "l1":
            self.regul = lambda X: torch.sum(torch.abs(X))

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
            print(sep + "\nInitialisation with Iterative PCA (Greed) ...")

            L0, X0 = init_estimate(self.const["science_data"], self.model.rot_angles, **kwargs)

            print("Done - running time : " + str(datetime.now() - start_time) + sep)

            if save:
                if not isdir(save): mkdir(save)
                print("Save init from in " + save + "...")
                write_fits(save + "/L0.fits", L0, verbose=False)
                write_fits(save + "/X0.fits", X0, verbose=False)

        # -- Define constantes
        self.L0x0 = (L0, X0)

        return L0, X0

    def estimate(self, w_r=0, maxiter=10, kactiv=0, save=False, gif=False, verbose=False):
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
        loss_evo = []

        # ______________________________________
        # Define constantes and convert arry to tensor

        science_data = torch.unsqueeze(torch.from_numpy(const["science_data"]), 1).double()
        science_data.requires_grad = False

        L0, X0 = self.L0x0

        L0 = self.model.mask * torch.unsqueeze(torch.from_numpy(L0), 0).double()


        X0 = self.model.mask * torch.unsqueeze(torch.from_numpy(X0), 0).double()

        # ______________________________________
        #  Optimizer definition  

        Lk, Xk = L0, X0
        optimizer = optim.LBFGS([Lk, Xk])
        Lk.requires_grad = True; Xk.requires_grad = True

        # Model and loss at step 0
        Y = self.model.forward_ADI(Lk, Xk)
        loss = self.fun_loss(Y, science_data)
        R = w_r * self.regul(Xk)

        def closure():
            nonlocal R
            nonlocal loss
            optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                Yk = self.model.forward_ADI(Lk, Xk)
                R =  w_r * self.regul(Xk)
                loss = self.fun_loss(Yk, science_data) + Ractiv * R
                loss.backward()
            return loss

        # ______________________________________
        #  Minimizations Start    

        start_time = datetime.now()
        print(sep + "\nResolving mayo optimization problem ..." +
                    "\nMinimiz LBFGS with '"+str(self.config[1])+"' loss and '"+str(self.config[0])+"' regul\n" +
                    "\nHypermarameter is set to {:.2f}".format(w_r)+", maxiter="+str(maxiter)+"\n")


        for k in range(maxiter):
            loss_evo.append(loss)
            if verbose: print("Iteration n°" + str(k) + " {:.6e}".format(loss) +
                                "\n\tWith R = {:.4e} ({:.0f}%)".format(R, 100*R/(loss)) +
                                 "\n\tand loss = {:.4e} ({:.0f}%) \n".format(loss, 100*(loss-R)/loss))
            if gif: print_iter(Lk, Xk, k, loss, R, self.config, w_r, kactiv)

            if k > kactiv: Ractiv = 1  # Only activate regul after few iterations
            optimizer.step(closure)

        print("Done - Running time : " + str(datetime.now() - start_time))

        # ______________________________________
        # Done, Store and unwrap results back to numpy array!

        L_est, X_est = abs(L0.detach().numpy()[0, :, :]), abs(X0.detach().numpy()[0, :, :])

        # Result dict
        res = {'state': optimizer.state,
               'x': (L_est, X_est),
               'loss_evo': loss_evo}

        self.res = res

        # Save
        if gif: iter_to_gif()
        if save: write_fits(save + "/L_est.fits", L_est), write_fits(save + "/X_est.fits", X_est)

        return L_est, X_est

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

    def constant_flux_repartition(self):
        if not hasattr(self, 'res'): raise AssertionError("Estimation should be lunched first")
        L_est, X_est = self.res['x']
        rad_L = []; rad_X = []
        rad_len = self.model.frame_shape[0]//2
        for raduis in range(1,rad_len):

            sel_point = np.linspace(np.pi /(4 * raduis), 2 * np.pi - np.pi /(4 * raduis), raduis * 4)

            y = raduis * np.cos(sel_point); x = raduis * np.sin(sel_point)

            y = np.rint(np.array([y[k]+0.5 if y[k]>0 else y[k]-0.5 for k in range(len(y))])).astype(int)
            x = np.rint(np.array([x[k]+0.5 if x[k]>0 else x[k]-0.5 for k in range(len(x))])).astype(int)

            y += rad_len; x += rad_len
            rad_L.append(np.min(L_est[x, y]))
            rad_X.append(np.min(X_est[x, y]))

            # Change const raduis from L to X
            X_est[x, y] += np.min(L_est[x, y])
            L_est[x, y] -= np.min(L_est[x, y])

        self.rad_L = rad_L;
        self.rad_X = rad_X
        return L_est, X_est

    def get_science_data(self):
        return self.model.rot_angles, self.model.phi_coro, self.const["science_data"]
