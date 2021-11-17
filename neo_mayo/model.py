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
from astropy.convolution import convolve_fft
import numpy as np
import torch
from torchvision.transforms.functional import rotate

# %% Forward ADI model : 

class model_ADI :
    """ Forward models as preented in mayo 
        
            Y = L + R(x)
        
        with L = starlight/constant flux in the image
             x = circumstellar flux
     """
    
    def __init__(self,rot_angles,phi_coro,mask,conv="astro",rot_opt=None):
        
        self.rot_angles = rot_angles # Frames known rotations list
        self.phi_coro   = phi_coro   # Reponse of the coronograph
        self.mask       = mask       # Pupil mask
        self.torch      = False
        
        self.nb_frame    = len(rot_angles)
        self.frame_shape = phi_coro.shape
        
        #-- Options 
        
        if rot_opt == "fastest":
            self.rot_args = {"imlib":'opencv',"interpolation":'lanczos4'}
        if rot_opt == None:
            self.rot_args = {}
            
        
        if conv == "astro" : 
            self.conv     = convolve_fft
            self.conv_arg = {"normalize_kernel":False,"nan_treatment":'fill'}
        if conv == "scipy" : 
            self.conv     = fftconvolve
            self.conv_arg = {"mode":"same"}

    def adapt_torch(self,delta):
        self.torch      = True
        self.phi_coro   = torch.from_numpy(self.phi_coro)  # Reponse of the coronograph
        self.mask       = torch.from_numpy(self.mask)      # Pupil mask
        self.hub_loss   = torch.nn.HuberLoss('sum',delta)

        
    def forward_ADI(self,L,x): 
        """ Process forward model as discribe in mayo : Y = M * ( L + conv(phi,R(x)) )  """
        
        Y = np.ndarray((self.nb_frame,) + L.shape)
        
        for frame_id in range(self.nb_frame) :
            Rx = frame_rotate(x,self.rot_angles[frame_id],**self.rot_args)
            Y[frame_id] = self.mask * ( L + self.conv(self.phi_coro, Rx,**self.conv_arg) )
        
        return Y
    
    def forward_torch_ADI(self,L,x): 
        """ Process forward model as discribe in mayo : Y = M * ( L + conv(phi,R(x)) )  """
        
        Y = torch.zeros((self.nb_frame,) + L.shape)
        
        for frame_id in range(self.nb_frame) :
            Rx = rotate(x,float(self.rot_angles[frame_id]))
            conv_Rx = torch.ifft( torch.fft(self.phi_coro) * torch.fft(Rx) ) 
            Y[frame_id] = self.mask * ( L + conv_Rx )
        
        return Y


# %% Constrain definitions 

def create_constrain_list(model,constantes):
    """ Create the list of model constraints """
    
    cons = []
    
    # Constrains on L (starlight)
    cons.append( {'type':'eq','fun':proj_L,'args':(model,constantes)} )
    
    return cons
    
def proj_L(var,model,constantes):  
    """ L contrain : Projection of L have to be equal to L"""
    
    L,x = var_inmatrix(var,model.frame_shape[0])

    proj = constantes["L_proj"]
    
    return np.sum((proj @ L) - L)
    

# %% Loss functions and wrappers 

from neo_mayo.utils import var_inmatrix,var_inline
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
    # Unwrap varaible
    L,x = var_inmatrix(var,model.frame_shape[0])
    
    loss = adi_model_loss(model,L,x,constantes) 
    
    if constantes["regul"] : loss += regul_L(L,constantes) + regul_X(x,constantes)
    
    return loss
    
                        
def adi_model_loss(model,L,x,constantes):
    """ ADI loss models as discribe in mayo
     Loss models of the model 
     
     loss = huberLoss( Y - (M * (L + conv(phi,R(x)) ) )
     
       Parameters
       ----------
       model : neo-mayo.model.model_ADI
           Forward model (M * (L + conv(phi,R(x)) ) )
        
        L,x : ndarray
            inputs
            
        constante : dict
            dictornay containing the non-varaible of the model :
            * science-data : science data inputs
            * delta : indicating the quadratic vs. linear loss changepoint of huber loss
        
       
       Returns
       -------
       loss : float
           Value of loss function (see neo_mayo.model.adi_model_loss)
       
    """
    
    # Unpack constantes
    science_data = constantes["science_data"]
    delta        = constantes["delta"]
    
    #  Compute forward model
    Y = model.forward_ADI(L,x)
    
    #Compute model loss with huber distance
    loss = huber(delta,Y - science_data)
    
    return np.sum(loss)

def adi_model_loss_torch(model,L,x,constantes):
    """  same as normal but tensor freidnly  """
    
    # Unpack constantes
    science_data = constantes["science_data"]
    
    #  Compute forward model
    Y = model.forward_torch_ADI(L,x)
    
    #Compute model loss with huber distance
    loss = model.hub_loss(Y,science_data)
    
    return loss

# %% Loss fonction ggradient

def call_loss_grad(var,model,constantes):
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
    # Unwrap varaible
    L,x = var_inmatrix(var,model.frame_shape[0])
    
    return var_inline(loss_grad(model,L,x,constantes))
    

def loss_grad(model,L,x,constantes): 
    """ Process backward model xich is a simple order 1 différential
    approximation gradient
    
    """
    # Store gradient 
    gradL = np.zeros(L.shape)
    gradX = np.zeros(x.shape)
    
    # delta to derive
    dL = np.zeros(L.shape)
    dX = np.zeros(x.shape)
    
    # We try each strings d_x, d_y to compute gradients
    for d_x in range(L.shape[0]) :
        for d_y in range(L.shape[0]) :
            
            # Grad de L
            dL[d_x,d_y] = 1
            gradL += adi_model_loss(model,dL,dX,constantes)
            dL[d_x,d_y] = 0
            
            # Grad de X
            dX[d_x,d_y] = 1
            gradX += adi_model_loss(model,dL,dX,constantes)
            dX[d_x,d_y] = 0
    
    return gradL,gradX


def regul_L(L,constantes):
    """ Loss function on L prior """
   
    proj = constantes["L_proj"]
    mu   = constantes["hyper_p"]
    
    A = proj @ L
    
    return mu * np.sum((A - L)**2)

def regul_X(x,constantes):
    """ Loss function on x prior """
    return 0
