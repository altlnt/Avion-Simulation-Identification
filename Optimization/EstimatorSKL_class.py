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
import pandas as pd

class ModelRegressor(BaseEstimator):  

    def __init__(self, Dict_variables=None,train_batch_size=1,n_epochs=1,fitting_strategy="scipy"):
        """
        Called when initializing the classifier
        """
        self.train_batch_size=train_batch_size
        self.fitting_strategy=fitting_strategy
        self.n_epochs=n_epochs

        self.MoteurPhysique=MoteurPhysique(called_from_opti=True)
        
        if Dict_variables!=None:
            self.start_Dict_variables=Dict_variables
            self.current_Dict_variables=Dict_variables
            
        else:
            self.start_Dict_variables=self.MoteurPhysique.Dict_variables
            self.current_Dict_variables=self.MoteurPhysique.Dict_variables
            
        self.MoteurPhysique.Dict_variables=self.start_Dict_variables
        
        self.x_train_batch=[]
        self.y_train_batch=[]
        
        self.current_train_score=0
        self.current_test_score=0
        
        
        
    def model(self, x_data):
        
        self.MoteurPhysique.speed=np.array([x_data["speed_%i"%(i)] for i in range(3)]).flatten()
        self.MoteurPhysique.q=np.array([x_data["q_%i"%(i)] for i in range(4)]).flatten()
        self.MoteurPhysique.omega=np.array([x_data["omega_%i"%(i)] for i in range(3)]).flatten()
        self.MoteurPhysique.R=tf3d.quaternions.quat2mat(self.MoteurPhysique.q).reshape((3,3))
        
        joystick_input=np.array([x_data['joystick_%i'%(i)] for i in range(4)]).flatten()
        
        
        
        self.MoteurPhysique.compute_dynamics(joystick_input,x_data['t'].values)
        
        d=np.r_[self.MoteurPhysique.forces,self.MoteurPhysique.torque]
        
        output=pd.DataFrame(data=d.reshape((1,6)),columns=['forces_0','forces_1','forces_2',
                                            'torque_0','torque_1','torque_2'])  
        return output


    def Dict_variables_to_X(self,Dict):
        
        # V=[]
        # for key in np.sort([i for i in self.start_Dict_variables.keys()]):
        #     [V.append(i) for i in np.array(Dict[key]).flatten()]
        # print(V)
        
        V=[i for key in np.sort([i for i in self.start_Dict_variables.keys()])  for i in np.array(Dict[key]).flatten()]
        # for key in np.sort([i for i in self.start_Dict_variables.keys()]):
        #     [V.append(i) for i in np.array(Dict[key]).flatten()]
        # print(V)
        return np.array(V)    
    
    def X_to_Dict_Variables(self,V):
        Dict={}
        
        counter=0
        for i in np.sort([i for i in self.start_Dict_variables.keys()]):
            L=len(np.array(self.start_Dict_variables[i]).flatten())
            S=np.array(self.start_Dict_variables[i]).shape
            Dict[i]=V[counter:counter+L].reshape(S)
            counter=counter+L
        return Dict
    
    def cost(self,X_params=None,usage="training"):
        
        if usage not in (["training","train_eval","test_eval"]):
            print('usage not in (["training","train_eval","test_eval"])')
            raise
        
        DictVariable_X=self.X_to_Dict_Variables(X_params) if (X_params is not None) else self.current_Dict_variables
        
        self.MoteurPhysique.Dict_Variables=DictVariable_X
        self.current_Dict_variables = self.MoteurPhysique.Dict_Variables
        # print(self.x_train_batch.iloc[[0]].head(),'\n\n')
        
        if usage=="training":
            
            used_x_batch=self.x_train_batch 
            used_y_batch=self.y_train_batch 

        elif usage=="train_eval":
            used_x_batch=self.x_train
            used_y_batch=self.y_train
            
        elif usage=="train_eval":
            used_x_batch=self.x_test 
            used_y_batch=self.y_test


        self.y_pred_batch=pd.concat([self.model(used_x_batch.iloc[[i]]) for i in range(len(self.x_train_batch))])
        
        # print(self.x_train_batch)
        # print("ypred batch\n\n",self.y_pred_batch,"\n\n",len(self.y_pred_batch))
        # print(self.y_train_batch,"\n\n",len(self.y_train_batch))
        
        error=used_y_batch.reset_index()-self.y_pred_batch.reset_index()
        error=error.drop(columns=["index"])
        sum_error=error.sum()
        sum_error_forces=np.sum([sum_error['forces_%i'%(i)]**2 for i in range(3)])/len(sum_error)
        sum_error_torque=np.sum([sum_error['torque_%i'%(i)]**2 for i in range(3)])/len(sum_error)
       
        # error_torque=
        cout_forces=1.0
        cout_torque=1.0
        C=cout_forces*sum_error_forces + cout_torque*sum_error_torque         
        return C
    
    def fit(self, X_train, Y_train, X_test=None, Y_test=None):

        self.x_train = X_train
        self.y_train = Y_train
        
        self.x_test = X_test
        self.y_test = Y_test
        
        for i in range(self.n_epochs):
            
            self.x_train_batch=[]
            self.y_train_batch=[]
            sample_nmbr=0

            while sample_nmbr<len(self.x_train-1):     
                
                self.x_train_batch.append(self.x_train.loc[[sample_nmbr]])
                self.y_train_batch.append(self.y_train.loc[[sample_nmbr]])
                sample_nmbr+=1
    
                if len(self.x_train_batch)==self.train_batch_size:
                    
                    "batch is full beginning opti"
                    
                    self.x_train_batch=pd.concat(self.x_train_batch)
                    self.y_train_batch=pd.concat(self.y_train_batch)                        
                    
                    if self.fitting_strategy=="scipy":
                        X0=self.Dict_variables_to_X(self.current_Dict_variables)
                        # print(X0)
                        res = minimize(self.cost,
                             x0=X0)
    
                        self.current_Dict_variables=self.X_to_Dict_Variables(res['x'])
                                            
                    self.x_train_batch=[]
                    self.y_train_batch=[]   
                        
            self.current_train_score=self.cost(usage="train_eval")
            
            if self.x_test!=None and self.y_test!=None:
                self.current_test_score=self.cost(usage="test_eval")

        return self


from Optimizer_class import  Optimizer
from Simulation.MoteurPhysique_class import MoteurPhysique

m=ModelRegressor()
o=Optimizer()
# print(o.raw_data.keys())
o.prepare_data()
# print('X_train',o.X_train.head(),'\n\n\n')

# m.x_train_batch=pd.concat([o.X_train.loc[[i]] for i in range(3)])
# m.y_train_batch=pd.concat([o.Y_train.loc[[i]] for i in range(3)])

# print(m.x_train_batch)
m.fit(o.X_train,o.Y_train)
