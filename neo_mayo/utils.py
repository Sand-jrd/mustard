#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:37:16 2021

______________________________

| Utils Fonction for neo_mayo  |
______________________________


@author: sand-jrd
"""
import numpy as np

# %% Syntax warpper for simplify minimiz  usage.

def var_inline(L,x):
    """ Wrap parameters as a vector to fit minimiz syntax  Parameters
    
    Parameters 
    ----------
     L,x : ndarray
         Varaible input of minimiz

     Returns
     -------
     x0 : array
         inline array of the concatened variables
     
  """
    if not ( len(L.shape) == 2 and len(x.shape) == 2 and x.shape == L.shape ): 
        raise ValueError('L and x are supposed to be matrix 2D of same dimentions')
    return np.concatenate((L.flatten(),x.flatten()), axis=None)

def var_inmatrix(M,size):
    """ Unwrap parameters from minimiz (wich is a big vector) into matrixs 
        Matching the order we wrapped it in var inline.
   
    Parameters
    ----------
    M : array
        Varaible taken from minimiz wich is an inline array of 
        concatened parameters
    
    size : int
        dimention of matrix
        
    nb_frame:  int
        number if frame
    
    Returns
    -------
    L,x : ndarray
        Unwrapped parameters
    
 """
    if M.size != (2 * size*size) :
        raise ValueError("length of vector doesn't match expeted value")

    mat_lenght = size*size
    
    # As we define in var_inline, L should be first then x
    L = M[:mat_lenght].reshape(size,size)
    x = M[mat_lenght:].reshape(size,size)

    return L,x


# %% Create patterns

def circle(shape,r,offset=0.5):
    """ Create circle of 1 in a 2D matrix of zeros"
       
       Parameters
       ----------
       shape : tuple
           shape x,y of the matrix
       
       r : float
           radius of the circle
       r (optional): float
           offset from the center
       
       Returns
       -------
       M : ndarray
           Zeros matrix with a circle filled with ones
       
    """
    M = np.zeros(shape)
    w,l = shape
    for x in range(0, w):
           for y in range(0, l):
               if  pow(x-(w/2) + offset ,2) + pow(y-(l/2) + offset,2) < pow(r,2):
                   M[x,y] = 1
    return M

# %% Manage files

import glob
from vip_hci.fits import open_fits
import json

def unpack_science_datadir(datadir):
    
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
    angles = open_fits(datadir + "/" + data_info["angles"],verbose=False)
    psf    = open_fits(datadir + "/" + data_info["psf"],verbose=False)
    if len(psf.shape) == 3 : psf[data_info["which_psf"]]
    
    science_data = open_fits(datadir + "/" + data_info["cube"],verbose=False)
    
    return angles,psf,science_data
