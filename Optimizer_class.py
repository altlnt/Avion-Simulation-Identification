#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:05:05 2021

@author: l3x
"""


import numpy as np
import os
import time
from datetime import datetime
import transforms3d as tf3d
from MoteurPhysique_class import MoteurPhysique
from scipy.optimize import minmize
import pandas as pd 


class Optimizer():
    
    
    def __init__(self):
        
        self.result_save_path=""
        self.log_file=""
        
        self.moteur_physique=MoteurPhysique()
        
        self.optimizer_mode="recursive"
        
        self.search_params=[]
        
        self.data=pd.read_csv(self.log_file)
        
        
        
        
    def init_optimization(self):
        
        
        
    def genstartpoint(self):
        return
    
    def cost(self):
        return
    
    def minimize(self):
        return
    
    def display_progress(self):
        return 
    
    
    
        