#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:05:05 2021

@author: l3x
"""
import sys
sys.path.append('../')

import numpy as np
import os
import time
from datetime import datetime
import transforms3d as tf3d
from Simulation.MoteurPhysique_class import MoteurPhysique
from scipy.optimize import minimize
import pandas as pd 
# from OptiMonitor_class import OptiMonitor
import regex as re

class Optimizer():
    
    
    def __init__(self):
        
        self.result_save_path="../OptiResults/"
        self.result_dir_name=""
        self.result_dir_name=self.result_dir_name if self.result_dir_name!="" else str(len(os.listdir(self.result_save_path))+1)
        
        os.makedirs(os.path.join(self.result_save_path,self.result_dir_name))
        
        
        
        self.log_dir_path="../Logs/2021_05_10_12h21m18s"
        self.log_path=os.path.join(self.log_dir_path,"log.txt")        
        self.true_params_path=os.path.join(self.log_dir_path,"log_params.txt")
        
        # self.moteur_physique=MoteurPhysique()
        
           
        
        self.raw_data=pd.read_csv(self.log_path)
        self.true_params={}
        
        with open(self.true_params_path, 'r') as f:
            s = f.readlines()
        split_list=[i.split("=") for i in s]
        # for i in split_list:
        #     key=i[0]
        #     # val=i[1].replace("\n","")
        #     print(key)
        
        print(split_list)
    # def init_optimization(self):
        
        
        
    # def genstartpoint(self):
    #     return
    
    # def cost(self):
    #     return
    
    # def minimize(self):
    #     return
    
    # def display_progress(self):
    #     return 
    
    
    
Optimizer()