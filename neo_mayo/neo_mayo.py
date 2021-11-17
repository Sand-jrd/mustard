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
from vip_hci.fits import open_fits,write_fits
from os import mkdir
from os.path import isdir 

# For verbose
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
sep = ('_' * 50) + "\n"

# Minimizers and its wrappers  
from scipy.optimize import minimize as scipy_minimiz
from optimparallel import minimize_parallel as parallel_minimiz
from utils import var_inmatrix,var_inline

#Algos and science model
from algo import init_estimate, compute_L_proj, torch_minimiz
from model import model_ADI,call_loss_function, adi_model_loss_torch, call_loss_grad, create_constrain_list

# Other
from neo_mayo.utils import circle


# %%


class mayo_estimator(): 
    """ Neo-mayo Algorithm main class 
   
    /!\ Neo-mayo isn't exaclty the same as Mayo pipeline. 
    Some choices have been re-thinked

    Parameters
    ----------
    datadir : str
        name of the data dir
        data dir should contained fits and a json file in wich it wich data info are 
        A json template is available in the exemple directory. 

    """
    
    def __init__(self,datadir="./data",mask_size=None):
        self.create_model_ADI(datadir,mask_size) 
        self.L0x0 = None
    
    def initalisation(self,from_dir=None,save=None):
        """ The first step with greed aim to find a good initialisation 
         
         Parameters
         ----------
         from_dir (optional) : str
             if None, compute iterative PCA
             if a path is given, get L0 and X0 from the directory
         save : str
             if a path is given, save L0 and X0 on the directory
             if None, do not save
         
         Returns
         -------
         L_est, X_est: ndarray
             Estimated starlight NON-DEROTATED (L) 
             and circunstlellar contributions DEROTATED (X) 
         
        """

        if from_dir :
            print("Get init from pre-processed datas...")
            L0 = open_fits(from_dir + "/L0.fits",verbose=False)
            X0 = open_fits(from_dir + "/X0.fits",verbose=False)
            
        else : 
            
            # -- Iterative PCA for varaible init -- #
            start_time = datetime.now()
            print(sep + "\nInitialisation with Iterative PCA (Greed) ...")
           
            L0,X0 = init_estimate(self.constantes["science_data"],self.model.rot_angles)
            
            print("Done - running time : " + str(datetime.now() - start_time) + sep)
           
            if save : 
                if not isdir(save) : mkdir(save)
                print("Get init from pre-processed datas...")
                write_fits(save + "/L0.fits", L0, verbose=False)
                write_fits(save + "/X0.fits", X0, verbose=False)
                
            
        # -- Define constantes
        self.L0x0 = var_inline(L0,X0)
        self.constantes["L_proj"] = compute_L_proj(L0)
        
        return L0, X0
        
    
    def estimate(self,delta=1e4,hyper_p=0,minimizer="minimize_parallel",**kwarg):
        """ Resovle the minimization problem as discribe in mayo
            The first step with greed aim to find a good initialisation 
            The second step process to the minimization
         
         Parameters
         ----------
         datadir : str
             path to the directory where are stored science data
             This directory shound contain a json file discribing its content
         
         delta : float
             indicating the quadratic vs. linear loss changepoint of huber loss
         
        hyper_p : float
             Hyperparameter to give accurate weight to regularization terme for L prior
         
        **kargs : dict
             minimize arguments
             (see scipy.optimize.minimize)
         
         Returns
         -------
         L_est, X_est: ndarray
             Estimated starlight (L) and circunstlellar (X) contributions
         
        """
        
        if self.L0x0 is None : raise AssertionError("No L0/x0. You need to run initialisation")
        
        

        
        # -- Define constantes 
        self.constantes["delta"]   = delta
        self.constantes["hyper_p"] = hyper_p
        
        # -- Default mode; can be changed in 'Options' below
        self.constantes["regul"]   = True
        fun = call_loss_function
        jac = call_loss_grad
        
        # ______________________________________
        # -- Options
        
        if minimizer == "minimize_parallel" : minimiz = parallel_minimiz
        
        if minimizer == "torch" : 
            minimiz = torch_minimiz
            fun = adi_model_loss_torch
            jac = None
            
        if minimizer == "L-BFGS-B" :
            minimiz = scipy_minimiz
            kwarg["method"] = "L-BFGS-B"
            
        if minimizer == "SLSQP" :
                minimiz = scipy_minimiz
                kwarg["method"] = "SLSQP"
                self.constantes["regul"]   = False
                kwarg["constraints"] = create_constrain_list(self.model,self.constantes)
        

        
        # ______________________________________
        #  Minimize considering mayo loss model    
        

        start_time = datetime.now()
        print(sep + "\nResolving mayo optimization problem ...")
        
        res = minimiz(fun  = fun,
                      jac  = jac,
                      x0   = self.L0x0,
                      args = (self.model,self.constantes), 
                      **kwarg)
        
        print("Done - Running time : " + str(datetime.now() - start_time))
        
        # ______________________________________
        # Done, Store and unwrap results !
        self.res = res
        L_est, X_est = var_inmatrix(res.x)
        
        return L_est, X_est
    
    


    # _____________________________________________________________
    # _____________ Tools fonctions of mayo_estimator _____________ 

    def create_model_ADI(self,datadir,mask_size):
        """ Initialisation of ADI models based on where the given data """
        
        angles,psf,science_data = unpack_science_datadir(datadir)
        
        # Set up a default pupil mask size based on the frame size
        if mask_size == None : mask_size = psf.shape[0]-10
        mask   = circle(psf.shape,mask_size)

        # Store science data as it is a constante
        self.constantes = {"science_data" : science_data}

        #  Init and return model ADI
        self.shape     = psf.shape
        self.nb_frames = science_data.shape[0]
        self.model     = model_ADI(angles,psf,mask)
    