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
from sklearn.model_selection import train_test_split



import pandas as pd 
# from OptiMonitor_class import OptiMonitor
import json



class Optimizer():
    
    
    def __init__(self):
        
        self.result_save_path="../OptiResults/"
        self.result_dir_name=""
        self.result_dir_name=self.result_dir_name if self.result_dir_name!="" else str(len(os.listdir(self.result_save_path))+1)
        
        os.makedirs(os.path.join(self.result_save_path,self.result_dir_name))
        
        
        
        self.log_dir_path="../Logs/2021_05_10_16h13m12s"
        self.log_path=os.path.join(self.log_dir_path,"log.txt")        
        self.true_params_path=os.path.join(self.log_dir_path,"params.json")
        
        with open(self.true_params_path,"r") as f:
            self.true_params=json.load( f)

        
        self.raw_data=pd.read_csv(self.log_path)
        # print(self.true_params)
        
        # self.moteur_physique=MoteurPhysique()

        print()
         
    def prepare_data(self):
        "drop alpha, t and omegadot from data"
        temp_df=self.raw_data.drop(columns=['alpha'])
        temp_df=temp_df.drop(columns=[i for i in temp_df.keys() if 'omegadot' in i])
        
        "renaming acc[0] and co to acc_0"
        for i in temp_df.keys():
            temp_df[i.replace('[','_').replace(']','')]=temp_df[i]
            temp_df=temp_df.drop(columns=[i])
        
        "accel at timestamp k+1 is computed using state at step k"
        
        new_temp_df=pd.DataFrame()
        
        for i in temp_df.keys():
            if ('acc' in i) or ('torque' in i):
                new_temp_df[i]=temp_df[i][1:].values
            else:
                new_temp_df[i]=temp_df[i][:-1].values
            
        self.data_prepared=new_temp_df
        
        "split between X and Y"

        self.data_prepared_train,self.data_prepared_test=train_test_split(self.data_prepared,test_size=0.2, random_state=42)
        

        self.X_train=self.data_prepared_train[[i for i in self.data_prepared.keys() if not (('acc' in i) or ('torque' in i))]]
        self.X_test=self.data_prepared_test[[i for i in self.data_prepared.keys() if not (('acc' in i) or ('torque' in i))]]
        self.Y_train=self.data_prepared_train[[i for i in self.data_prepared.keys() if (('acc' in i) or ('torque' in i))]]
        self.Y_test=self.data_prepared_train[[i for i in self.data_prepared.keys() if (('acc' in i) or ('torque' in i))]]


        return        
    # def init_optimization(self):
        
        
        
    # def genstartpoint(self):
    #     return
    
    # def cost(self):
    #     return
    
    # def minimize(self):
    #     return
    
    # def display_progress(self):
    #     return 
    
    
    
O=Optimizer()
O.prepare_data()

print(O.raw_data,O.X_train,O.Y_train)
print(O.X_train.keys())