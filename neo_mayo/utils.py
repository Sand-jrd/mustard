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
    """ Wrap parameters as a vector to fit minimiz syntax """
    if len(L.shape) == 3 and len(x.shape) == 3 and x.shape == L.shape: 
        raise ValueError('L and x are supposed to be cubes 3D of same dimentions')
    return np.concatenate((L.flatten(),x.flatten()), axis=None)

def var_inmatrix(M,size,nb_frames):
    """ Unwrap parameters from minimiz (wich is a big vector) into matrixs 
    Matching the order we wrapped it in var inline.
    """
    if M.size != (2 * size**2 * nb_frames) :
        raise ValueError('L and x are supposed to be cubes 3D of same dimentions')
    
    mat_lenght = size*2 * nb_frames
    
    # As we define in var_inline, L should be first then x
    L = M[:mat_lenght].reshape(nb_frames,size,size)
    x = M[mat_lenght:].reshape(nb_frames,size,size)

    return [L,x]
