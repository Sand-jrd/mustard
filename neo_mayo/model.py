#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:42:10 2021

______________________________

| Forward model used in mayo |
______________________________

@author: sand-jrd
"""

from vip_hci.preproc import frame_rotate
import torch  
import numpy as np


# %% Forward ADI model : 

class model_ADI :
    """ Forward models as preented in mayo 
        
            Y = L + R(x)
        
        with L = starlight/constant flux in the image
             x = circumstellar flux
     """
    
    def __init__(self,rot_angles,phi_coro,mask):
        
        self.rot_angles = rot_angles # Frames known rotations list
        self.phi_coro   = phi_coro   # Reponse of the coronograph
        self.mask       = mask        # Pupil mask
        
    def forward_ADI(self,L,x): 
        """ Process forward model as discribe in mayo : Y = M * ( L + conv(phi,R(x)) )  """
        return self.mask( L + np.convolve(self.phi_coro * frame_rotate(x,self.rot_angles) , mode='same' ) )

                        
def adi_model_loss():
    """ ADI loss models as discribe in mayo
     Loss models if the function we want to minimiz """
    
    return None