#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:33:38 2021

Usage axemple of neo-mayo

@author: sand-jrd
"""

from neo_mayo import mayo_estimator

# %% Test mayo estimator initialisation

# Choose where to get datas
datadir = "./example-data"

# init the estimator
estimator = mayo_estimator(datadir)


# %% Test Forward model

import numpy as np
from neo_mayo.utils import circle

shape = estimator.shape
model = estimator.model

L = circle((256,256),5)
x = circle((256,256),20) - circle((256,256),30,offset=0)

L = np.stack((L,L,L,L))
x = np.stack((x,x,x,x))

Y = model.forward_ADI(L,x)

# =============================================================================
# # Show results
# 
# import matplotlib.pyplot as plt
# 
# plt.figure("Forward model")
# for frame_ID in range(Y.shape[0]) : 
#     plt.subplot(4,2,frame_ID+1),plt.imshow(Y[frame_ID], cmap = "jet")
#     
# plt.subplot(4,2,5),plt.imshow(L[frame_ID], cmap = "jet")
# plt.subplot(4,2,6),plt.imshow(x[frame_ID], cmap = "jet")
# =============================================================================

# %% Test Estimations

estimator.estimate()
