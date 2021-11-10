#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:30:10 2021

______________________________

|      Neo-Mayo estimator       |
______________________________


@author: sand-jrd
"""


import numpy as np
from neo_mayo.utils import unpack_science_datadir,circle

# For verbose
sep = ('_' * 50) + "\n"
from datetime import datetime

# Minimize and its wrappers  
from scipy.optimize import minimize
from utils import var_inmatrix,var_inline

#Algos and science model
from algo import Greed, compute_L_proj
from model import model_ADI,call_loss_function


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
    
    def __init__(self,datadir = "./data", mask_size = None):
        self.create_model_ADI(datadir,mask_size) 
    
    def estimate(self,delta=1e4,hyper_p=0,**kwarg):
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
        kwarg["method"] = "L-BFGS-B"
        
        self.constantes["delta"]   = delta
        self.constantes["hyper_p"] = hyper_p
        # ______________________________________
        # Step one : Initialisation

        # -- Greed for varaible init  
        start_time = datetime.now()
        print(sep + "\nInitialisation with Iterative PCA (Greed) ...")
       
        L0,X0 = Greed(self.constantes["science_data"],self.model.rot_angles)
        
        print("Done - running time : " + str(datetime.now() - start_time) + sep)
        
        # -- Define constantes
        self.constantes["L_proj"] = compute_L_proj(L0)
        
        
        # ______________________________________
        # Step two : Minimize considering mayo loss model    
        

        start_time = datetime.now()
        print(sep + "\nResolving mayo optimization problem ...")
        
        res = minimize(fun  = call_loss_function,
                       x0   = var_inline(L0,X0),
                       args = (self.model,self.constantes), 
                       **kwarg)
        
        print("Done - Running time : " + str(datetime.now() - start_time) + sep)
        
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
        self.model     =  model_ADI(angles,psf,mask)
    