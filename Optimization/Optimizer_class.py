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
import time
import numpy as np

class Optimizer():
    
    
    def __init__(self):
        

        self.log_dir_path="../Logs/2021_06_02_16h45m15s"
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
            if i not in ('t','takeoff'):
                temp_df=temp_df.drop(columns=[i])
        
        "accel at timestamp k+1 is computed using state at step k"
        # print(temp_df.keys())
        new_temp_df=pd.DataFrame()
        
        for i in temp_df.keys():
            if ('forces' in i) or ('torque' in i) or ("joystick" in i) or (i in ('t')):
                new_temp_df[i]=temp_df[i][1:].values
            else:
                new_temp_df[i]=temp_df[i][:-1].values
            
        self.data_prepared=new_temp_df
        
        "split between X and Y"

        self.data_prepared_train,self.data_prepared_test=train_test_split(self.data_prepared,test_size=0.1, random_state=41)
        
        self.data_prepared_train,self.data_prepared_test=self.data_prepared_train.reset_index(),self.data_prepared_test.reset_index()



        self.X_train=self.data_prepared_train[[i for i in self.data_prepared.keys() if not (('forces' in i) or ('torque' in i))]]
        self.X_test=self.data_prepared_test[[i for i in self.data_prepared.keys() if not (('forces' in i) or ('torque' in i))]]
        self.Y_train=self.data_prepared_train[[i for i in self.data_prepared.keys() if (('forces' in i) or ('torque' in i))]]
        self.Y_test=self.data_prepared_test[[i for i in self.data_prepared.keys() if (('forces' in i) or ('torque' in i))]]

        print([(i,j) for (i,j) in enumerate(self.X_train.keys())],[(i,j) for (i,j) in enumerate(self.Y_train.keys())])



        self.X_train=self.X_train.values
        self.X_test=self.X_test.values
        self.Y_train=self.Y_train.values
        self.Y_test=self.Y_test.values


    
        return      

    def launch(self):
        self.prepare_data()
        # print(self.estimator.start_Dict_variables,"\n")
        # print(self.estimator.current_Dict_variables,"\n")
        # self.estimator.x_train,self.estimator.y_train=self.X_train,self.Y_train

        # for i in range(3):
        #     print(self.estimator.cost(usage="train_eval"))

        # for i in self.estimator.start_Dict_variables.keys():
        #     print(self.estimator.start_Dict_variables[i])
        #     print(self.estimator.current_Dict_variables[i])
        #     print(self.estimator.MoteurPhysique.Dict_variables[i],"\n")

            
        # self.estimator.monitor.update()
        # print("X_train",self.X_train)
        # print("X_test",self.X_test)
        
        self.estimator.generate_random_params(amp_dev=0.1)
        
        
        self.estimator.x_train=self.X_train
        self.estimator.y_train=self.Y_train
        self.estimator.x_test=self.X_test
        self.estimator.y_test=self.Y_test
        self.estimator.x_train_batch=self.estimator.x_train
        self.estimator.y_train_batch=self.estimator.y_train
        # ti=time.time()
        self.estimator.cost(usage="train_eval")
        # X0params=self.estimator.Dict_variables_to_X(self.estimator.real_Dict_variables)
        # G=self.estimator.compute_gradient(self.estimator.cost,X0params,eps=1e-8,gradfunc=None)
        # print(G)
        # print("X: ",self.X_test.shape, "Y", self.Y_test.shape)
        # print()
        # print(time.time()-ti)
        
        # self.estimator.generate_random_params(amp_dev=3)
        # print(self.estimator.cost(usage="train_eval"))
        # X0_params=self.estimator.Dict_variables_to_X(self.estimator.current_Dict_variables)

        # self.estimator.compute_gradient(self.estimator.cost,X0_params,eps=1e-7)

        t1 = time.time()
        self.estimator.fit(self.X_train,self.Y_train,self.X_test,self.Y_test)
        print(time.time()-t1)

        
        
O=Optimizer()
# import time
O.launch()
