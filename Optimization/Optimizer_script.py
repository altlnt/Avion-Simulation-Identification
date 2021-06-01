#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 22:41:50 2021

@author: l3x
"""

import sys
sys.path.append('../')

import os
from sklearn.model_selection import train_test_split
import pandas as pd 
import json
import time
import numpy as np
from Simulation.MoteurPhysique_class import MoteurPhysique as MPHI
from OptiMonitor_class import OptiMonitor_MPL
import transforms3d as tf3d
from scipy.optimize import minimize

log_dir_path="../Logs/2021_05_31_11h29m30s"
log_path=os.path.join(log_dir_path,"log.txt")        
true_params_path=os.path.join(log_dir_path,"params.json")

with open(true_params_path,"r") as f:
    true_params=json.load( f)
raw_data=pd.read_csv(log_path)
    
result_save_path="../OptiResults/"
result_dir_name=""
result_dir_name=result_dir_name if result_dir_name!="" else str(len(os.listdir(result_save_path))+1)

os.makedirs(os.path.join(result_save_path,result_dir_name))


spath=os.path.join(result_save_path,result_dir_name)

train_batch_size=5
fitting_strategy="custom_gradient"
n_epochs=10
learning_rate=1e-4

MoteurPhysique=MPHI(called_from_opti=True)
opti_variables_keys=['alpha_stall',
                          'largeur_stall',
                          'cd1sa',
                          'cl1sa',
                          'cd0sa',
                          'cd1fp',
                          'cd0fp',
                          'coeff_drag_shift',
                          'coeff_lift_shift',
                          'coeff_lift_gain']
        

if 'Dict_variable' in locals() or 'Dict_variable' in globals():
    start_Dict_variables=Dict_variables
    real_Dict_variables=Dict_variables
    current_Dict_variables=Dict_variables
    
else:
    start_Dict_variables=MoteurPhysique.Dict_variables
    current_Dict_variables=MoteurPhysique.Dict_variables
    real_Dict_variables=MoteurPhysique.Dict_variables
    
MoteurPhysique.Dict_variables=start_Dict_variables

x_train_batch=[]
y_train_batch=[]

current_epoch=0
current_train_score=0
current_test_score=0
sample_nmbr=0
simulator_called=0


# monitor=OptiMonitor_MPL()
# monitor.t=current_epoch
# monitor.y_train,monitor.y_eval= current_train_score,current_test_score
# monitor.init_params=start_Dict_variables
# monitor.current_params=start_Dict_variables
# monitor.update()


gradient_kp=1.0
gradient_kd=0.0
gradient_ki=0.0
gradient_integ_lim=-1.0
gradient_func=None

########################### params 

temp_df=raw_data.drop(columns=['alpha'])
temp_df=temp_df.drop(columns=[i for i in temp_df.keys() if 'omegadot' in i])

"renaming acc[0] and co to acc_0"
for i in temp_df.keys():
    temp_df[i.replace('[','_').replace(']','')]=temp_df[i]
    if i not in ('t','takeoff'):
        temp_df=temp_df.drop(columns=[i])

"accel at timestamp k+1 is computed using state at step k"

new_temp_df=pd.DataFrame()

for i in temp_df.keys():
    if ('forces' in i) or ('torque' in i) or ("joystick" in i) or (i in ('t')):
        new_temp_df[i]=temp_df[i][1:].values
    else:
        new_temp_df[i]=temp_df[i][:-1].values
    
data_prepared=new_temp_df

"split between X and Y"

data_prepared_train,data_prepared_test=train_test_split(data_prepared,test_size=0.1, random_state=41)

data_prepared_train,data_prepared_test=data_prepared_train.reset_index(),data_prepared_test.reset_index()



X_train=data_prepared_train[[i for i in data_prepared.keys() if not (('forces' in i) or ('torque' in i))]]
X_test=data_prepared_test[[i for i in data_prepared.keys() if not (('forces' in i) or ('torque' in i))]]
Y_train=data_prepared_train[[i for i in data_prepared.keys() if (('forces' in i) or ('torque' in i))]]
Y_test=data_prepared_test[[i for i in data_prepared.keys() if (('forces' in i) or ('torque' in i))]]

X_train=X_train.values
X_test=X_test.values
Y_train=Y_train.values
Y_test=Y_test.values

########################### funcs

def Dict_variables_to_X(Dict,opti_variables_keys=opti_variables_keys):
    V=[j  for key in np.sort([i for i in opti_variables_keys]) for j in np.array(Dict[key]).flatten()]
    return np.array(V)    

def X_to_Dict_Variables(V,
                        opti_variables_keys=opti_variables_keys,
                        start_Dict_variables=start_Dict_variables):
    Dict={}
    
    counter=0
    for i in np.sort([i for i in opti_variables_keys]):
        L=len(np.array(start_Dict_variables[i]).flatten())
        S=np.array(start_Dict_variables[i]).shape
        Dict[i]=V[counter:counter+L].reshape(S)
        counter=counter+L
    for i in start_Dict_variables.keys():
        if i not in opti_variables_keys:
            Dict[i]=start_Dict_variables[i]
    return Dict

def generate_random_params(X_params,amp_dev=0.0,verbose=True):
    
    new_X_params=X_params*(1+amp_dev*(np.random.random(size=len(X_params))-0.5))    

    print('\n[Randomization] X Old / X New variables: ')
    for i,j in zip(X_params,new_X_params):
        print(i,j,"diff = %f %%"%(round(abs(i-j)/i*100.0,2)))
    
    return new_X_params


def model(X_params, x_data):

    
    t,takeoff=x_data[0],x_data[1]
    
    speed=x_data[5:8]
    omega=x_data[11:14]
    q=x_data[14:18]
    joystick_input=x_data[18:]
    
    MoteurPhysique.speed=speed
    MoteurPhysique.q=q
    MoteurPhysique.omega=omega
    MoteurPhysique.R=tf3d.quaternions.quat2mat(MoteurPhysique.q).reshape((3,3))
    MoteurPhysique.takeoff=takeoff
    
    MoteurPhysique.Dict_variables=X_to_Dict_Variables(X_params)
    print( MoteurPhysique.Dict_variables)
    MoteurPhysique.compute_dynamics(joystick_input,t)
    d=np.r_[MoteurPhysique.forces,MoteurPhysique.torque]
    
    output=d.reshape((1,6))
    
    return output



def cost(X_params,x_data,y_data,verbose=False):
    
    
    DictVariable_X=X_to_Dict_Variables(X_params) 
    MoteurPhysique.Dict_variables=DictVariable_X

    used_x_batch=x_data
    used_y_batch=y_data
    
    y_pred_batch=np.vstack([model(used_x_batch[i]) for i in range(len(used_x_batch))])

    
    y_pred_batch_error_sq=(used_y_batch-y_pred_batch)**2
    

    y_pred_batch_error_dict={}
    y_pred_batch_error_dict['sum_forces']=y_pred_batch_error_sq[0]
    y_pred_batch_error_dict['sum_forces']+=y_pred_batch_error_sq[1]
    y_pred_batch_error_dict['sum_forces']+=y_pred_batch_error_sq[2]
    
    y_pred_batch_error_dict['sum_torques']=y_pred_batch_error_sq[3]
    y_pred_batch_error_dict['sum_torques']+=y_pred_batch_error_sq[4]
    y_pred_batch_error_dict['sum_torques']+=y_pred_batch_error_sq[5]
    
 
    sum_error_forces=y_pred_batch_error_dict['sum_forces']/len(y_pred_batch_error_sq)

    sum_error_torque=y_pred_batch_error_dict['sum_torques']/len(y_pred_batch_error_sq)

    cout_forces=1.0
    cout_torque=1.0
    C=cout_forces*np.sum(sum_error_forces) + cout_torque*np.sum(sum_error_torque)  
  
    return C
        

def compute_numeric_gradient(func,X_params,eps=1e-6,verbose=False):
        grad=[0 for k in range(len(X_params))]
        for i in range(len(X_params)):
            f1 = func(X_params+np.array([eps if j==i else 0 for j in range(len(X_params))]))
            f2 = func(X_params-np.array([eps if j==i else 0 for j in range(len(X_params))]))
            grad[i]= (f1 - f2)/(2*eps)

        return grad

def compute_symbolic_gradient(X_params,x_data):

        MoteurPhysique.Dict_variables=X_to_Dict_Variables(X_params)
        MoteurPhysique.Theta = opti_variables_keys
        Gradien_results = []
        y_pred_batch=np.vstack([model(X_params,x_data[i]) for i in range(len(x_data))])

        for i in range(len(x_data)):
            
            t,takeoff=x_data[i,0],x_data[i,1]
            
            speed=x_data[i,5:8]
            omega=x_data[i,11:14]
            q=x_data[i,14:18]
            joystick_input=x_data[i,18:]

            MoteurPhysique.speed=speed
            MoteurPhysique.q=q
            MoteurPhysique.omega=omega
            MoteurPhysique.R=tf3d.quaternions.quat2mat(MoteurPhysique.q).reshape((3,3))
            MoteurPhysique.takeoff=takeoff
            MoteurPhysique.compute_dynamics(joystick_input, t , compute_gradF=True)  
            Gradien_results.append(np.r_[MoteurPhysique.grad_forces,MoteurPhysique.grad_torque])
            
        " grad = -2 *(y_data-y_pred) * gradient "
        gradbatch=[(-2.0*(y_train[i]-y_pred_batch[i])@Gradien_results[i])\
                    for i in range(len(y_pred_batch))]
        gradbatch=np.array([i.reshape((len(X_params),)) for i in gradbatch])
        grad = np.sum(gradbatch[i] for i in range(len(gradbatch)))/ len(gradbatch)
        
            
        return grad
    
x_train = X_train
y_train = Y_train

x_test = X_test
y_test = Y_test

current_train_score=-1.0


nnD=X_to_Dict_Variables(Dict_variables_to_X(MoteurPhysique.Dict_variables))


for i in nnD.keys():
    if i not in MoteurPhysique.Dict_variables.keys() :
        print("in last not in first:",i)

for i in MoteurPhysique.Dict_variables.keys() :
    if i not in nnD.keys():
        print("in first not in last:",i)


for i in range(n_epochs):
    "saving"
    
    if spath is not None:
        with open(os.path.join(spath,"%i.json"%(i)),'w') as f:
            sdict={}
            for i in opti_variables_keys:
                sdict[i]=current_Dict_variables[i]
                
            sdict['train_score']=current_train_score
            sdict['test_score']=current_test_score
            
            for i in sdict.keys():
                if type(sdict[i])==list:
                    sdict[i]=np.array(sdict[i]) 
                    
                if type(sdict[i])==np.ndarray:
                    sdict[i]=sdict[i].tolist() 
                    

            json.dump(sdict,f)
           

    "opti loop"
    x_train_batch=[]
    y_train_batch=[]           

    sample_nmbr=0
    current_epoch+=1

    while sample_nmbr<(len(x_train)-1):     
        
        x_train_batch.append(x_train[sample_nmbr])
        y_train_batch.append(y_train[sample_nmbr])
        sample_nmbr+=1

        if len(x_train_batch)==train_batch_size or (sample_nmbr==len(x_train)-1):
            
            "batch is full beginning opti"
            
            x_train_batch=np.vstack(x_train_batch)
            y_train_batch=np.vstack(y_train_batch)                        
            previous_Dict_variables=current_Dict_variables

            if fitting_strategy=="scipy":
                scaler=Dict_variables_to_X(start_Dict_variables)
                scaler=np.array([i if i!=0 else 1.0 for i in scaler ])
                X0_params=Dict_variables_to_X(current_Dict_variables)

                
                res = minimize(cost,method='SLSQP',
                     x0=X0_params,options={'maxiter': 100})

                current_Dict_variables=X_to_Dict_Variables(res['x']*scaler)
                
            if fitting_strategy=="custom_gradient":
                
                scaler=Dict_variables_to_X(start_Dict_variables)
                scaler=np.array([i if i!=0 else 1.0 for i in scaler ])

                X0_params=Dict_variables_to_X(current_Dict_variables)

                
                G=compute_symbolic_gradient(X0_params,
                                            x_train_batch)

                new_X=X0_params-learning_rate*G
                current_Dict_variables=X_to_Dict_Variables(new_X)

            
            for i in opti_variables_keys:
                
                 print('########################\nreal/start/prev/current '+i+' :',
                      real_Dict_variables[i],
                      start_Dict_variables[i],
                      previous_Dict_variables[i],
                      current_Dict_variables[i])
                
            print('########################')
            x_train_batch=[]
            y_train_batch=[]   
            # input("Continue ?")
            # current_train_score=cost(usage="train_eval",verbose=True)
            # monitor.current_params=current_Dict_variables

            # monitor.update()
            
            # if x_test is not None and y_test is not None:
            #     current_test_score=cost(usage="test_eval",verbose=True)

