#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:05:05 2021

@author: l3x
"""
import sys
sys.path.append('../')

import os
from sklearn.model_selection import train_test_split
from EstimatorSKL_class import ModelRegressor
import pandas as pd 
import json



class Optimizer():
    
    
    def __init__(self):
        

        self.log_dir_path="../Logs/2021_05_10_16h13m12s"
        self.log_path=os.path.join(self.log_dir_path,"log.txt")        
        self.true_params_path=os.path.join(self.log_dir_path,"params.json")
        
        with open(self.true_params_path,"r") as f:
            self.true_params=json.load( f)
        self.raw_data=pd.read_csv(self.log_path)
            
        self.result_save_path="../OptiResults/"
        self.result_dir_name=""
        self.result_dir_name=self.result_dir_name if self.result_dir_name!="" else str(len(os.listdir(self.result_save_path))+1)
        
        os.makedirs(os.path.join(self.result_save_path,self.result_dir_name))
        
        self.estimator=ModelRegressor()
        self.estimator.spath=os.path.join(self.result_save_path,self.result_dir_name)
        
        
    def prepare_data(self):
        "drop alpha, t and omegadot from data"
        temp_df=self.raw_data.drop(columns=['alpha'])
        temp_df=temp_df.drop(columns=[i for i in temp_df.keys() if 'omegadot' in i])
        # print(temp_df.keys())
        "renaming acc[0] and co to acc_0"
        for i in temp_df.keys():
            temp_df[i.replace('[','_').replace(']','')]=temp_df[i]
            if i!='t':
                temp_df=temp_df.drop(columns=[i])
        
        "accel at timestamp k+1 is computed using state at step k"
        # print(temp_df.keys())
        new_temp_df=pd.DataFrame()
        
        for i in temp_df.keys():
            if ('forces' in i) or ('torque' in i):
                new_temp_df[i]=temp_df[i][1:].values
            else:
                new_temp_df[i]=temp_df[i][:-1].values
            
        self.data_prepared=new_temp_df
        
        "split between X and Y"

        self.data_prepared_train,self.data_prepared_test=train_test_split(self.data_prepared,test_size=0.2, random_state=42)
        
        self.data_prepared_train,self.data_prepared_test=self.data_prepared_train.reset_index(),self.data_prepared_test.reset_index()

        self.X_train=self.data_prepared_train[[i for i in self.data_prepared.keys() if not (('forces' in i) or ('torque' in i))]]
        self.X_test=self.data_prepared_test[[i for i in self.data_prepared.keys() if not (('forces' in i) or ('torque' in i))]]
        self.Y_train=self.data_prepared_train[[i for i in self.data_prepared.keys() if (('forces' in i) or ('torque' in i))]]
        self.Y_test=self.data_prepared_train[[i for i in self.data_prepared.keys() if (('forces' in i) or ('torque' in i))]]

    
        return      

    def launch(self):
        self.prepare_data()
        self.estimator.generate_random_params()
        self.estimator.monitor.update()
        self.estimator.fit(self.X_train,self.Y_train,self.X_test,self.Y_test)
        
        
        
O=Optimizer()
import time
O.launch()
