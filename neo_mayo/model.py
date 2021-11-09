#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:42:10 2021

______________________________

| Forward model used in mayo |
______________________________

@author: sand-jrd
"""


# %% Forward ADI model : 

from vip_hci.preproc import frame_rotate
import torch  

class model_ADI :
    """ Forward models as preented in mayo 
        
            Y = L + R(x)
        
        with L = starlight/constant flux in the image
             x = circumstellar flux
     """
    
    def __inti__(self,rot_angles):
        self.rot_angles = rot_angles
        
    def forward_ADI(self,L,x): 
        """ Process forward model as discribe in mayo : Y = L + R(x) """
        return L + frame_rotate(x,self.rot_angles)

    
def ADI_model_loss():
    """ ADI loss models as discribe in mayo
    Loss models if the function we want to minimiz """
    return None