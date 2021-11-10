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

# To manage files
import glob
from vip_hci.fits import open_fits
import json

# Math algo and model 
from scipy.optimize import minimize
from utils import var_inmatrix,var_inline,circle
from algo import Greed
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
    
    def __init__(self,datadir = "./data", mask_size = None, **kwarg):
        # TODO define parameters that mayo needs to work.
        self.minimz_param = kwarg
        self.minimz_param["method"] = "L-BFGS-B"
        
        self.create_model_ADI(datadir,mask_size) 
    
    def estimate(self,delta=1e4,init="zeros"):
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
         init : str
             Mode of initialisation of the minimization problem
             * "Greed" : compute Iterative PCA as init 
             * "zeros" : init with matrix of zeros
         
         Returns
         -------
         L_est, X_est: ndarray
             Estimated starlight (L) and circunstlellar (X) contributions
         
        """
        
        self.constantes["delta"] = delta
        
        # Step one : Find a good init with Greed.
        if init == "Greed" : 
            [L0,X0] = Greed()
        else : 
            L0 = X0 = np.zeros(( (self.nb_frames,) + self.shape) )
        
        # Step two : Minimize considering mayo loss model    
        res = minimize(fun  = call_loss_function,
                       x0   = var_inline(L0,X0),
                       args = (self.model,self.constantes), 
                       **self.minimz_param)
        
        
        # Store and unwrap results
        self.res = res
        L_est, X_est = var_inmatrix(res.x)
        
        return L_est, X_est
    
    
    
    
    
    
    # _____________________________________________________________
    # _____________ Tools fonctions of mayo_estimator _____________ 

    def create_model_ADI(self,datadir,mask_size):
        """ Initialisation of ADI models based on where the given data """
        
        #  Import data
        json_file = glob.glob(datadir + "/*.json")
        
        if len(json_file) == 0  : raise AssertionError("Json file not found in in data folder : "        + str(datadir))
        elif len(json_file) > 1 : raise AssertionError("More than two json file found in data folder : " + str(datadir))
        
        with open(json_file[0], 'r') as read_data_info:
            data_info = json.load(read_data_info)        
        
        # Checks if all required keys are here 
        required_keys = ("cube","angles","psf")
        if not all([key in data_info.keys() for key in required_keys]):
            raise AssertionError("Data json info does not contained required keys")

        # Open fits
        angles = open_fits(datadir + "/" + data_info["angles"])
        psf    = open_fits(datadir + "/" + data_info["psf"])
        if len(psf.shape) == 3 : psf[data_info["which_psf"]]
        
        # Set up a default pupil mask size based on the frame size
        if mask_size == None : mask_size = psf.shape[0]-10
        mask   = circle(psf.shape,mask_size)

        # Store science data as it is a constante
        science_data = open_fits(datadir + "/" + data_info["cube"])
        self.constantes = {"science_data" : science_data}

        #  Init and return model ADI
        self.shape     = psf.shape
        self.nb_frames = science_data.shape[0]
        self.model     =  model_ADI(angles,psf,mask)
    