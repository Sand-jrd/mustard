#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:30:10 2021

______________________________

|      Neo-Mayo estimator       |
______________________________


@author: sand-jrd
"""

from scipy.optimize import minimize
from utils import var_inmatrix,var_inline,circle
from algo import Greed
from model import model_ADI
from vip_hci.fits import open_fits
import glob
import json

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
        self.minimz_param["method"] = "BFGS"
        
        self.create_model_ADI(datadir,mask_size) 
           
    
    def estimate(self):
        """ Resovle the minimization problem as discribe in mayo
            The first step with greed aim to find a good initialisation 
            The second step process to the minimization
        """
        # Step one : Find a good init with Greed.
        [L0,X0] = Greed()
        
        # Step two : Minimize considering mayo loss model    
        res = minimize(None,var_inline(L0,X0),**self.minimz_param)
        
        [L_est,X_est] = var_inmatrix(res.x)
        
        return [L_est,X_est]
    
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

        angles = open_fits(datadir + "/" + data_info["angles"])
        psf    = open_fits(datadir + "/" + data_info["psf"])
        if len(psf.shape) == 3 : psf[data_info["which_psf"]]
        
        if mask_size == None : mask_size = psf.shape[0]-10
        mask   = circle(psf.shape,mask_size)

        #  Init and return model ADI
        self.shape = psf.shape
        self.model =  model_ADI(angles,psf,mask)
    