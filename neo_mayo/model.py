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
from scipy.signal import fftconvolve
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
        self.mask       = mask       # Pupil mask
        
        self.nb_frame    = len(rot_angles)
        self.frame_shape = phi_coro.shape

        
    def forward_ADI(self,L,x): 
        """ Process forward model as discribe in mayo : Y = M * ( L + conv(phi,R(x)) )  """
        Y = np.ndarray(L.shape)
        for frame_id in range(L.shape[0]) :
            Rx = frame_rotate(x[frame_id],self.rot_angles[frame_id])
            Y[frame_id] = self.mask * ( L[frame_id] + fftconvolve(self.phi_coro, Rx ,mode='same') )
        return Y

# %% Loss functions and wrappers 

from neo_mayo.utils import var_inmatrix
from scipy.special import huber

def call_loss_function(var,model,constantes):
    """
       Unwrap minimiz parameters and call adi model loss
       
       Parameters
       ----------
       var : array
           inline varaibles (matching the minimiz syntaxs)
       
       model : neo-mayo.model.model_ADI
           Forward model
       
       Returns
       -------
       loss : float
           Value of loss function (see neo_mayo.model.adi_model_loss)
       
    """
    L,x = var_inmatrix(var,model.frame_shape[0],model.nb_frame)
    return adi_model_loss(model,L,x,constantes)
    
                        
def adi_model_loss(model,L,x,constantes):
    """ ADI loss models as discribe in mayo
     Loss models if the function we want to minimiz """
    
    # Unpack constantes
    science_data = constantes["science_data"]
    delta        = constantes["delta"]

    #  Compute forward model
    Y = model.forward_ADI(L,x)
    
    #Compute model loss with huber distance
    loss = huber(delta,Y - science_data)
    
    return np.sum(loss)