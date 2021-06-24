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
import dill as dill
import sys 
from pylab import sort 

log_dir_path=os.path.join("../Logs/",sort(os.listdir("../Logs/"))[-1])
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


MoteurPhysique=MPHI(called_from_opti=True)
if not dill.load(open('../Simulation/function_moteur_physique','rb'))[-1]==None:
    opti_variables_keys=dill.load(open('../Simulation/function_moteur_physique','rb'))[-1]
else:
    print("Attention aux chargement des params pour l'identif")
    opti_variables_keys=['alpha0',
                         'cd1sa',
                          'cl1sa',
                          'cd0sa',
                          'coeff_drag_shift',
                          'coeff_lift_shift',
                          'coeff_lift_gain']

if not true_params==None:
    real_Dict_variables={i : true_params[i] for i in MoteurPhysique.Dict_variables.keys()}
    start_Dict_variables = real_Dict_variables
    MoteurPhysique.Dict_world={i : np.array(true_params[i]) for i in MoteurPhysique.Dict_world.keys() }
else:
    print("Error load true params")

x_train_batch=[]
y_train_batch=[]

current_epoch=0
current_train_score=0
current_test_score=0
sample_nmbr=0
simulator_called=0


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

data_prepared_train,data_prepared_test=train_test_split(data_prepared,test_size=100, random_state=41)

data_prepared_train,data_prepared_test=data_prepared_train.reset_index(),data_prepared_test.reset_index()



X_train=data_prepared_train[[i for i in data_prepared.keys() if not (('forces' in i) or ('torque' in i))]]
X_test=data_prepared_test[[i for i in data_prepared.keys() if not (('forces' in i) or ('torque' in i))]]
Y_train=data_prepared_train[[i for i in data_prepared.keys() if (('forces' in i) or ('torque' in i))]]
Y_test=data_prepared_test[[i for i in data_prepared.keys() if (('forces' in i) or ('torque' in i))]]

X_train=X_train.values
X_test=X_test.values
Y_train=Y_train.values
Y_test=Y_test.values

### Préparation des données non randomizé pour comparaison en 
X_test_sim = data_prepared.reset_index()[[i for i in data_prepared.keys() if not (('forces' in i) or ('torque' in i))]].values
Y_test_sim = data_prepared.reset_index()[[i for i in data_prepared.keys() if (('forces' in i) or ('torque' in i))]].values

########################### funcs

def Dict_variables_to_X(Dict,opti_variables_keys=opti_variables_keys):
    V=[j  for key in np.sort([i for i in opti_variables_keys]) for j in np.array(Dict[key]).flatten()]
    return np.array(V)    

def X_to_Dict_Variables(V, opti_variables_keys=opti_variables_keys, start_Dict_variables=start_Dict_variables):
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
    for params in new_X_params:
        if params <=0:
            params=0.005

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
    
    if type(X_params)==dict:
        MoteurPhysique.Dict_variables=X_params
    else:
        MoteurPhysique.Dict_variables=X_to_Dict_Variables(X_params)

    # print( MoteurPhysique.Dict_variables)
    MoteurPhysique.compute_dynamics(joystick_input,t)
    d=np.r_[MoteurPhysique.forces,MoteurPhysique.torque]
    
    output=d.reshape((1,6))
    
    return output



def cost(X_params,x_data,y_data,verbose=False, RMS=None):
    
    if type(X_params)==dict:
        DictVariable_X = X_params
    else:
        DictVariable_X=X_to_Dict_Variables(X_params) 

    MoteurPhysique.Dict_variables=DictVariable_X

    used_x_batch=x_data
    used_y_batch=y_data
    y_pred_batch=[]
    y_pred_batch=np.vstack([model(X_params, used_x_batch[i]) for i in range(len(used_x_batch))])

    
    y_pred_batch_error_sq=(used_y_batch-y_pred_batch)**2

    y_pred_batch_error_dict={}
    y_pred_batch_error_dict['sum_forces']=y_pred_batch_error_sq[0][0]
    y_pred_batch_error_dict['sum_forces']+=y_pred_batch_error_sq[0][1]
    y_pred_batch_error_dict['sum_forces']+=y_pred_batch_error_sq[0][2]
    
    y_pred_batch_error_dict['sum_torques']=y_pred_batch_error_sq[0][3]
    y_pred_batch_error_dict['sum_torques']+=y_pred_batch_error_sq[0][4]
    y_pred_batch_error_dict['sum_torques']+=y_pred_batch_error_sq[0][5]
 
    sum_error_forces=y_pred_batch_error_dict['sum_forces']/len(y_pred_batch_error_sq)

    sum_error_torque=y_pred_batch_error_dict['sum_torques']/len(y_pred_batch_error_sq)

    cout_forces=1.0
    cout_torque=1.0
    C=cout_forces*np.sum(sum_error_forces) + cout_torque*np.sum(sum_error_torque)  
    # if verbose==True:
    #     print("Epoch "+str(current_epoch)+" sample "+str(sample_nmbr) + "/" +str(len(x_data))+" "+usage+' cost : '+str(C))
    if not RMS==None:
        return C, np.sum(sum_error_forces) , np.sum(sum_error_torque)
    else:
        return C
        

def compute_numeric_gradient(func,X_params, x_data, y_data, eps=1e-6,verbose=False):
        grad=[0 for k in range(len(X_params))]
        for i in range(len(X_params)):
            f1 = func(X_params+np.array([eps if j==i else 0 for j in range(len(X_params))]), x_data, y_data)
            f2 = func(X_params-np.array([eps if j==i else 0 for j in range(len(X_params))]), x_data, y_data)
            grad[i]= (f1 - f2)/(2*eps) 
        grad = [i*len(x_data) for i in grad]
        if not np.linalg.norm(grad)==0:
            return grad/np.linalg.norm(grad)
        else:
            return grad

def compute_symbolic_gradient(X_params,x_data, y_data):

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
        gradbatch=[(-2.0*(y_data[i]-y_pred_batch[i])@Gradien_results[i])\
                    for i in range(len(y_pred_batch))]
        gradbatch=np.array([i.reshape((len(X_params),)) for i in gradbatch])
        grad = sum(gradbatch[i] for i in range(len(gradbatch)))/ len(gradbatch) 
        if not np.linalg.norm(grad)==0:
            return grad/ np.linalg.norm(grad)
        else:
            return grad


### PID gradient
gradient_kp=1
gradient_kd=10*0
gradient_ki=0.01*0
gradient_integ_lim=-1.0
gradient_func=None
G_sum =0
G =10
####### Params pour l'opti ######
n_params_dropout = 2
train_batch_size=20
fitting_strategy="custom_gradient"
n_epochs=30
learning_rate=0.5e-3
list_learning_rate=[learning_rate, learning_rate*50000, learning_rate*80, learning_rate*200, learning_rate*100, learning_rate*100]
name=("Opti avec Learning rate initial = "+str(format(learning_rate, '.1E'))+" Avec PID = "+str(gradient_kp)+"/" +str(gradient_ki) + "/" + str(gradient_kd)+" avec des batch de taille :"+ str(train_batch_size))

########## Initialisation des paramètres #########
x_train = X_train
y_train = Y_train

x_test = X_test
y_test = Y_test

current_train_score=1.
start_Dict_variables = X_to_Dict_Variables(generate_random_params\
                    (Dict_variables_to_X(real_Dict_variables),amp_dev=1,verbose=True))
current_Dict_variables =start_Dict_variables
# current_Dict_variables['cd0sa']=true_params['cd0sa']
# current_Dict_variables['cl1sa']=true_params['cl1sa']
# current_Dict_variables['cd1sa']=true_params['cd1sa']
# current_Dict_variables['coeff_lift_shift']=true_params['coeff_lift_shift']

best_Dict_variables = current_Dict_variables

######### Création du monitor ##########
monitor=OptiMonitor_MPL(name,opti_variables_keys=opti_variables_keys, params_real=true_params)
monitor.x_data=data_prepared['t'].values     # Temps de la simulation pour la comparaison y_sim/y_real
monitor.y_train,monitor.y_eval= current_train_score,current_test_score
monitor.init_params=start_Dict_variables
monitor.current_params=start_Dict_variables
monitor.y_sim = [model(current_Dict_variables, X_test_sim[i]) for i in range(len(X_test_sim))]
current_test_score, monitor.RMS_forces, monitor.RMS_torque =cost(current_Dict_variables, x_test, y_test, verbose=True, RMS=True)
monitor.y_real= Y_test_sim        # Valeur réel des efforts 
b=0
monitor.current_params_to_opti =list(start_Dict_variables.keys())[b]
# monitor.update(epoch=True)

############################
# nnD=X_to_Dict_Variables(Dict_variables_to_X(MoteurPhysique.Dict_variables))


# for i in nnD.keys():
#     if i not in MoteurPhysique.Dict_variables.keys() :
#         print("in last not in first:",i)

# for i in MoteurPhysique.Dict_variables.keys() :
#     if i not in nnD.keys():
#         print("in first not in last:",i)

t2 = time.time()
z=0 # Compteur pour l'affichage des simulations en fonctions des epochs 
############ Début de l'optimization ##########

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

    monitor.t = current_epoch
    monitor.y_eval, monitor.y_train = current_test_score,current_train_score
    monitor.update(epoch=True)
    n_update_sim=(n_epochs)/3
    if monitor.t+1%n_update_sim==0:
        monitor.update_sim_monitor(n_epoch=z)
        z+=1
    elif monitor.t==0:
        monitor.update_sim_monitor(n_epoch=z)
        z+=1
    elif monitor.t+1==n_epochs:
        monitor.update_sim_monitor(n_epoch=z)
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

                
                res = minimize(cost(X0_params, x_train_batch, y_train_batch),method='SLSQP',
                      x0=X0_params,options={'maxiter': 100})

                current_Dict_variables=X_to_Dict_Variables(res['x']*scaler)
                
            if fitting_strategy=="custom_gradient":
                
                # scaler=Dict_variables_to_X(start_Dict_variables)
                # scaler=np.array([i if i!=0 else 1.0 for i in scaler ])

                X0_params=Dict_variables_to_X(current_Dict_variables)
                G_pred = G

                G=compute_symbolic_gradient(X0_params,
                                            x_train_batch, y_train_batch)
                G_dot= (G_pred-G)
                G_sum+=G
                G_total = (gradient_kp * G + gradient_ki * G_sum + gradient_kd * G_dot)
                # if not n_params_dropout==0:
                #     for lzar in range(n_params_dropout):
                #         kir=np.random.randint(0,len(X0_params))
                #         G_total[kir]= 0
              
                for o,val in enumerate(G_total):
                    if not o==b:
                        G_total[o]=0
                new_X=X0_params-learning_rate*G_total
                current_Dict_variables=X_to_Dict_Variables(new_X)
                
            print('########################')
            x_train_batch=[]
            y_train_batch=[]   
            # input("Continue ?")
            current_train_score=cost(current_Dict_variables, x_train, y_train, verbose=True)
            print("epoch" ,current_epoch, "\n sample_nmbr = " , sample_nmbr,"/",len(x_train), "\n current_train_score = ", current_train_score)
            monitor.current_params=current_Dict_variables
            monitor.sample=sample_nmbr
            
            if x_test is not None and y_test is not None:
                new_test_score, monitor.RMS_forces, monitor.RMS_torque =cost(current_Dict_variables, x_test, y_test, verbose=True, RMS=True)
                if new_test_score<current_test_score:
                    best_Dict_variables=current_Dict_variables
                current_test_score=new_test_score
            
            monitor.update()
            if monitor.list_params_finish:
                for keys in monitor.list_params_finish.keys():
                    if keys==monitor.current_params_to_opti:
                        b+=1
                        if b>len(opti_variables_keys)-1:
                            b=0
                            monitor.list_params_finish={}
                        monitor.current_params_to_opti=opti_variables_keys[b]
                        learning_rate=list_learning_rate[b]
    monitor.y_sim = [model(current_Dict_variables, X_test_sim[i]) for i in range(len(X_test_sim))]
    monitor.y_real=Y_test_sim

                
print(time.time()-t2)
for i in opti_variables_keys:
  
    print('########################\n real/start/current '+i+' :',
        real_Dict_variables[i],
        start_Dict_variables[i],
        best_Dict_variables[i])