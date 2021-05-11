#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:21:48 2021

@author: l3x
"""

import sys
sys.path.append('../')

import numpy as np
from sklearn.base import BaseEstimator
import transforms3d as tf3d
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
import pandas as pd

class ModelRegressor(BaseEstimator):  

    def __init__(self, Dict_variables=None,train_batch_size=1,fitting_strategy="scipy"):
        """
        Called when initializing the classifier
        """
        self.train_batch_size=train_batch_size
        self.fitting_strategy=fitting_strategy
        self.MoteurPhysique=MoteurPhysique()
        
        if Dict_variables!=None:
            self.start_Dict_variables=Dict_variables
            self.current_Dict_variables=Dict_variables
            
        else:
            self.start_Dict_variables=self.MoteurPhysique.Dict_variables
            self.current_Dict_variables=self.MoteurPhysique.Dict_variables
            
        self.MoteurPhysique.Dict_variables=self.start_Dict_variables
    


    def _model(self, x):
        
        self.MoteurPhysique.speed=np.array([x["speed_%i"%(i)] for i in range(3)])
        self.MoteurPhysique.q=np.array([x["q_%i"%(i)] for i in range(4)])
        self.MoteurPhysique.omega=np.array([x["omega_%i"%(i)] for i in range(3)])
        self.MoteurPhysique.R=tf3d.quaternions.quat2mat(self.MoteurPhysique.q)
        
        joystick_input=np.array([x['joystick_%i'%(i)] for i in range(4)])
        self.MoteurPhysique.compute_dynamics(joystick_input,x['t'])
        d=np.r_[self.MoteurPhysique.forces,self.MoteurPhysique.torque]
        output=pd.DataFrame(data=d.reshape((1,6)),columns=['forces_0','forces_1','forces_2',
                                            'torque_0','torque_1','torque_2'])  
        return output 


    def Dict_variables_to_X(self,Dict):
        return
    
    def Dict_variables_to_X(self,Dict):
        return    
    
    def fit(self, X, Y):

        self.x_train = X
        self.y_train = Y
        
        sample_nmbr=0
        x_train_batch=[]
        y_train_batch=[]
        
        while sample_nmbr<len(self.x_train-1):     
            
            x_train_batch.append(self.x_train.loc[sample_nmbr])
            y_train_batch.append(self.y_train.loc[sample_nmbr])
            sample_nmbr+=1

            if len(x_train_batch==self.train_batch_size):
                
                "beginning opti"
                if self.fitting_strategy=="scipy":
                    
            
                
                    
        
        res = minimize(
            self._f,
            x0=param_initial_values, 
        )
        print("Optimization result:\n\n",res)
        self.param_a,self.param_b = res['x']
        return self


from Optimizer_class import  Optimizer
from Simulation.MoteurPhysique_class import MoteurPhysique

m=ModelRegressor()
o=Optimizer()
# print(o.raw_data.keys())
o.prepare_data()
# print(o.X_train.keys())
print(m._model(o.X_train.loc[0]))

a=np.arange(20)

print([i for i in gen_batches(20,1)])
