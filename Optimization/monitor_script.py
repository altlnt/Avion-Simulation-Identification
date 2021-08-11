#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:41:02 2021

@author: mehdi
"""
import os
import csv
import matplotlib.pyplot as plt
import dill as dill
from pylab import sort
import pandas as pd

log_name = sort(os.listdir("/home/mehdi/Documents/identification_modele_avion/OptiResults/Opti_real/"))[-3]


opti_path_result = os.path.join('/home/mehdi/Documents/identification_modele_avion/OptiResults/Opti_real/'+log_name)
with open(opti_path_result+'/results.csv') as data:
    reader = csv.reader(data)
    dic = pd.read_csv(data)
opti_variables_keys=dic.keys()

fig = plt.figure()
list_fig=[]
for i in range(len(opti_variables_keys)):
    list_fig.append(fig.add_subplot(3,3,i+1))
     
n=0
fig.suptitle('optimisation '+log_name)
for keys, val in dic.items():
    list_fig[n].plot(val)
    list_fig[n].set_ylabel(keys)
    list_fig[n].set_xlabel('sample')
    n+=1
    
    
    
    
    
    
    
    
    