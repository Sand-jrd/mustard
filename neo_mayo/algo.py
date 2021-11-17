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
from vip_hci.itpca import pca_it

import numpy as np

# For verbose
from datetime import datetime

# %% Iterative PCA aka GREED

def init_estimate(cube, angle_list,method = None,max_comp=1,nb_iter=10,**kwarg):
    """ Iterative PCA as describe in Mayo """
    
    L_k = np.zeros(cube.shape)
    X_k = np.zeros(cube.shape)
    nb_frame = cube.shape[0]
    
    if method == "sand" : 

        for iter_k in range(nb_iter):
            for nb_comp in range(max_comp): 
                
                res = pca_fullfr.pca(L_k, angle_list,ncomp=nb_comp,verbose=False)
               
                # Since res is derotated we have to rerotate it ...
                for frame_id in range(nb_frame) : 
                    frame_id_rot   = frame_rotate(res,angle_list[frame_id]) 
                    L_k[frame_id]  = cube[frame_id] - abs(frame_id_rot)
                    X_k[frame_id]  = frame_id_rot
                    
    else : 
        
        res = pca_it(cube, angle_list,verbose=False,**kwarg)
        
        for frame_id in range(nb_frame) : 
            frame_id_rot   = frame_rotate(res,angle_list[frame_id]) 
            L_k[frame_id]  = cube[frame_id] - abs(frame_id_rot)    
    
    return np.median(L_k,axis=0),res

# %% Minimizer with toch

import torch
import torch.optim as optim
from neo_mayo.utils import var_inmatrix,var_inline

def torch_minimiz(fun,x0,args,nb_iter=10,**kwarg) : 
  
    model       = args[0]
    constantes  = args[1]  
    model.adapt_torch(constantes["delta"])
    L0,X0 = var_inmatrix(x0,model.frame_shape[0])
    
    L0 = torch.from_numpy(L0)
    L0.requires_grad = True
    
    X0 = torch.from_numpy(X0)
    X0.requires_grad = True

    optimizer = optim.LBFGS([L0,X0],
                            line_search_fn="strong_wolfe")
    
    for ii in range(nb_iter):
        optimizer.zero_grad()
        objective = fun(model,L0,X0,constantes)
        objective.backward()
        optimizer.step(lambda: fun(model,L0,X0,constantes))
        
    dict_res = optimizer.state["defaultdict"]
    for key in dict_res.keys() : 
        res = dict_res[key]
    
    res['x'] = var_inline(L0,X0)
    return res

# %% Other ALGO
 

def compute_L_proj(L_est,rank=1):
    """ Compute a L projector """
    
    U,S,V = np.linalg.svd(L_est)
    proj = U @ np.transpose(U)
    
    return proj