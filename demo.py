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

shape = estimator.shape
model = estimator.model

L = np.zeros(shape)
x = np.zeros(shape)

model.forward_ADI(L,x)