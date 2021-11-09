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
from utils import var_inmatrix,var_inline
from algo import Greed
from model import ADI_model_loss

class mayo_estimator(): 
    """ Neo-mayo Algorithm main class 
   
    /!\ Neo-mayo isn't exaclty the same as Mayo pipeline. 
    Some choices have been re-thinked

    """
    
    def __init__(self,**kwarg):
        # TODO define parameters that mayo needs to work.
        self.minimz_param = kwarg
        self.minimz_param["method"] = "BFGS"
    
    def estimate(self):
        
        # Step one : Find a good init with Greed.
        # TODO
        [L0,X0] = Greed()
        
        # Step two : Minimize considering mayo loss model    
        # TODO
        res = minimize(ADI_model_loss,var_inline(L0,X0),**self.minimz_param)
        
        [L_est,X_est] = var_inmatrix(res.x)
        
        return [L_est,X_est]