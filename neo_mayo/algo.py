#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:27:42 2021

______________________________

| algortihms used in neo-mayo |
______________________________

@author: sand-jrd
"""

from vip_hci.pca import pca_fullfr 
from vip_hci.preproc import frame_rotate
import numpy as np

# %% Iterative PCA aka GREED

def Greed(cube, angle_list,max_comp=2,nb_iter=10):
    """ Iterative PCA as describe in Mayo """
    
    L_k      = np.zeros(cube.shape)
    X_k      = np.zeros(cube.shape)

    nb_frame = cube.shape[0]
    
    for iter_k in range(nb_iter):
        for nb_comp in range(max_comp): 
            
            res = pca_fullfr.pca(L_k, angle_list,ncomp=nb_comp,verbose=False)
           
            # Since res is derotated we have to rerotate it ...
            for frame_id in range(nb_frame) : 
                frame_id_rot   = frame_rotate(res,angle_list[frame_id]) 
                L_k[frame_id]  = cube[frame_id] - frame_id_rot
                X_k[frame_id]  = frame_id_rot

    
    X_est = X_k
    L_est = L_k
    
    return L_est,X_est

# %% Other ALGO
 

def compute_L_proj(cube,rank=1):
    """ Compute a L projector """
    
    U,S,V = np.linalg.svd(cube)
    
    if len(U.shape) == 3 :
        frame_k = 0
        proj = U[frame_k] @ np.transpose(U[frame_k])

        for frame_k in range(1,U.shape[0]-1) : 
            np.stack( (proj, U[frame_k] @ np.transpose(U[frame_k])) )
    
    return proj