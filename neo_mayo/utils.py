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
    if not ( len(L.shape) == 3 and len(x.shape) == 3 and x.shape == L.shape ): 
        raise ValueError('L and x are supposed to be cubes 3D of same dimentions')
    return np.concatenate((L.flatten(),x.flatten()), axis=None)

def var_inmatrix(M,size,nb_frames):
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
    if M.size != (2 * size**2 * nb_frames) :
        raise ValueError('L and x are supposed to be cubes 3D of same dimentions')
    
    mat_lenght = size**2 * nb_frames
    
    # As we define in var_inline, L should be first then x
    L = M[:mat_lenght].reshape(nb_frames,size,size)
    x = M[mat_lenght:].reshape(nb_frames,size,size)

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