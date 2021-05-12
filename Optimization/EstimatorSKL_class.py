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
import json
import os
from OptiMonitor_class import OptiMonitor_MPL
from Simulation.MoteurPhysique_class import MoteurPhysique
import time

class ModelRegressor(BaseEstimator):  

    def __init__(self, Dict_variables=None,train_batch_size=10,n_epochs=3,fitting_strategy="scipy"):

        self.spath=None
        
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
        
        self.current_epoch=0
        self.current_train_score=0
        self.current_test_score=0
        
        self.monitor=OptiMonitor_MPL()
        self.monitor.t=self.current_epoch
        self.monitor.y_train,self.monitor.y_eval= self.current_train_score,self.current_test_score
        self.monitor.update(self.current_Dict_variables)
        self.sample_nmbr=0
        return 
    
    def generate_random_params(self,amp_dev=0.0):
        
        X_params=self.Dict_variables_to_X(self.start_Dict_variables)
        new_X_params=X_params*(1+amp_dev*(np.random.random(size=len(X_params))-0.5))
        self.current_Dict_variables=self.X_to_Dict_Variables(new_X_params)
        return

    def Dict_variables_to_X(self,Dict):
        V=[i for key in np.sort([i for i in self.start_Dict_variables.keys()])  for i in np.array(Dict[key]).flatten()]
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


    
    def cost(self,X_params=None,usage="training"):
        
        if usage not in (["training","train_eval","test_eval"]):
            print('usage not in (["training","train_eval","test_eval"])')
            raise
        
        DictVariable_X=self.X_to_Dict_Variables(X_params) if (X_params is not None) else self.current_Dict_variables
        
        self.MoteurPhysique.Dict_variables=DictVariable_X
        self.current_Dict_variables = self.MoteurPhysique.Dict_variables
        # print(self.x_train_batch.iloc[[0]].head(),'\n\n')
        
        if usage=="training":
            
            used_x_batch=self.x_train_batch 
            used_y_batch=self.y_train_batch 

        elif usage=="train_eval":
            # print(self.x_train_batch ,self.x_train)
            used_x_batch=self.x_train
            used_y_batch=self.y_train
            
            
        elif usage=="test_eval":
            used_x_batch=self.x_test 
            used_y_batch=self.y_test

        # print(len(used_x_batch),usage)
        self.y_pred_batch=pd.concat([self.model(used_x_batch.iloc[[i]]) for i in range(len(used_x_batch))])
        
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
        
        print("Epoch "+str(self.current_epoch)+" sample "+str(self.sample_nmbr) + "/" +str(len(self.x_train))+' cost : '+str(C))

        return C
    
    def fit(self, X_train, Y_train, X_test=None, Y_test=None):

        self.x_train = X_train
        self.y_train = Y_train
        
        self.x_test = X_test
        self.y_test = Y_test
        
        self.current_train_score=self.cost(usage="train_eval")
        
        if self.x_test is not None and self.y_test is not None:
            self.current_test_score=self.cost(usage="test_eval")
            
        self.monitor.t=self.current_epoch
        self.monitor.y_train,self.monitor.y_eval= self.current_train_score,self.current_test_score
        self.monitor.update(self.current_Dict_variables)
        
        for i in range(self.n_epochs):
            "saving"
            
            if self.spath is not None:
                with open(os.path.join(self.spath,"%i.json"%(i)),'w') as f:
                    sdict=self.current_Dict_variables
                    sdict['train_score']=self.current_train_score
                    sdict['test_score']=self.current_test_score
                    
                    for i in sdict.keys():
                        if type(sdict[i])==list:
                            sdict[i]=np.array(sdict[i]) 
                            
                        if type(sdict[i])==np.ndarray:
                            sdict[i]=sdict[i].tolist() 
                            
                        # print(sdict)
                        
                    json.dump(sdict,f)
                   
            "monitor update"
            self.monitor.t=self.current_epoch
            self.monitor.y_train,self.monitor.y_eval= self.current_train_score,self.current_test_score
            self.monitor.update(self.current_Dict_variables)
            print(self.current_Dict_variables.keys())
            "opti loop"
            self.x_train_batch=[]
            self.y_train_batch=[]           


            self.sample_nmbr=0
            self.current_epoch+=1

            while self.sample_nmbr<(len(self.x_train)-1):     
                
                self.x_train_batch.append(self.x_train.loc[[self.sample_nmbr]])
                self.y_train_batch.append(self.y_train.loc[[self.sample_nmbr]])
                self.sample_nmbr+=1
    
                if len(self.x_train_batch)==self.train_batch_size or (self.sample_nmbr==len(self.x_train)-1):
                
                    
                    "batch is full beginning opti"
                    
                    self.x_train_batch=pd.concat(self.x_train_batch)
                    self.y_train_batch=pd.concat(self.y_train_batch)                        

                    if self.fitting_strategy=="scipy":
                        X0_params=self.Dict_variables_to_X(self.current_Dict_variables)
                        # print(X0)
                        res = minimize(self.cost,method='SLSQP',
                             x0=X0_params,options={'maxiter': 2})
                        print("finires")
                        self.current_Dict_variables=self.X_to_Dict_Variables(res['x'])
                                            
                    self.x_train_batch=[]
                    self.y_train_batch=[]   
                        
            self.current_train_score=self.cost(usage="train_eval")
            
            if self.x_test is not None and self.y_test is not None:
                self.current_test_score=self.cost(usage="test_eval")


                    
                    
        return self


# from Optimizer_class import  Optimizer
# from Simulation.MoteurPhysique_class import MoteurPhysique

# m=ModelRegressor()
# o=Optimizer()
# print(o.raw_data.keys())
# o.prepare_data()
# print('X_train',o.X_train.head(),'\n\n\n')

# m.x_train_batch=pd.concat([o.X_train.loc[[i]] for i in range(3)])
# m.y_train_batch=pd.concat([o.Y_train.loc[[i]] for i in range(3)])

# print(m.x_train_batch)
# print(m.start_Dict_variables)
# nd=m.X_to_Dict_Variables(m.Dict_variables_to_X(m.start_Dict_variables))

# for i in m.start_Dict_variables.keys():
#     print("ORIGINAL : ",m.start_Dict_variables[i])
#     print("RECONSTRUCTED : ",nd[i],"\n")

    
