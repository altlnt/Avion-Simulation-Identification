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
import numpy as np
from Simulation.MoteurPhysique_class import MoteurPhysique as MPHI
from OptiMonitor_class import OptiMonitor_MPL
import transforms3d as tf3d
from scipy.optimize import minimize
import dill as dill
import sys 
from pylab import sort 
from datetime import datetime
import csv

def main_func(x):
    log_real=x[0]
    plot=x[1]
    log_name=x[2]
    name_script='parallele_opti.py'
    train_batch_size=x[3] 
    n_epochs=1000
    learning_rate=x[4]
    fitting_strategy=x[5]
    wind_X=x[6][0]
    wind_Y=x[6][1]
    type_grad=x[7]
    
    num=x[-1]
    # %% Initialisation de l'opti en fonction des données d'entrées (réelles ou simulées) et créations des noms des dossier

    if log_real==True:
        plot=False
        log_path=os.path.join('/home/mehdi/Documents/identification_modele_avion/Logs/log_real/'+log_name+'/log_real.csv')     
        
        "#### Chargement des données depuis le fichier de log, et initialisation de l'optimizeur, attention"
        "les noms des variables peuvent être trompeur quand on fait une optimization avec des données réelles"
        "#### Ex : true_params ne signifie pas les paramètres réels, mais seulement les paramètres initiaux de l'opti"
        true_params     =  {"wind_X" :wind_X,  \
                            "wind_Y" :wind_Y,  \
                            "wind_Z" :0,  \
                            "g"    : np.array([0,0,9.81]),                    \
                            "mode" : [1,1,1,1],\
                            "masse": 8.5 , \
                            "inertie": np.diag([1.38,0.84,2.17]),\
                            "aire" : [0.62*0.262* 1.292 * 0.5, 0.62*0.262* 1.292 * 0.5, 0.34*0.1* 1.292 * 0.5, 0.34*0.1* 1.292 * 0.5, 1.08*0.31* 1.292 * 0.5],\
                            "cp_list": [np.array([-0.013,0.475,-0.040],       dtype=np.float).flatten(), \
                                        np.array([-0.013,-0.475,-0.040],      dtype=np.float).flatten(), \
                                        np.array([-1.006,0.17,-0.134],    dtype=np.float).flatten(),\
                                        np.array([-1.006,-0.17,-0.134],   dtype=np.float).flatten(),\
                                        np.array([0.021,0,-0.064],          dtype=np.float).flatten()],
                            "alpha0" : np.array([0.07,0.07,0,0,0.07]),\
                            "alpha_stall" : 0.3391428111 ,                     \
                            "largeur_stall" : 15.0*np.pi/180,                  \
                            "cd0sa" : 0.010,\
                            "cd0fp" : 0.010,\
                            "cd1sa" : 4.55, \
                            "cl1sa" : 5, \
                            "cd1fp" : 2.5, \
                            "coeff_drag_shift": 0.5, \
                            "coeff_lift_shift": 0.05, \
                            "coeff_lift_gain": 2.5,\
                            "Ct": 1.1e-4, \
                            "Cq": 1e-8, \
                            "Ch": 1e-4,\
                            "rotor_moy_speed":925/2}
            
        raw_data=pd.read_csv(log_path)
        
        result_save_path="../OptiResults/Opti_real"
        result_dir_name="identification_"+datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss_numero_")+str(num)
        result_dir_name=result_dir_name if result_dir_name!="" else str(len(os.listdir(result_save_path))+1)
    
    else:
        log_dir_path=os.path.join("/home/mehdi/Documents/identification_modele_avion/Logs/log_sim",\
                                  sort(os.listdir("/home/mehdi/Documents/identification_modele_avion/Logs/log_sim"))[-1])
        log_path=os.path.join(log_dir_path,"log.txt")     
        true_params_path=os.path.join(log_dir_path,"params.json")
        
        "#### Chargement des données de simulation, et initialisation de l'optimizeur"
        with open(true_params_path,"r") as f:
            print("chargement des vraies données de simu...")
            true_params=json.load(f)
            print("Chargement réussi : \n " + str(true_params))
            raw_data=pd.read_csv(log_path)
        result_save_path="../OptiResults/Opti_sim"
        result_dir_name="identification_"+datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
        result_dir_name=result_dir_name if result_dir_name!="" else str(len(os.listdir(result_save_path))+1)
# %% Création des répertoires pour sauvegarder les données

    os.makedirs(os.path.join(result_save_path,result_dir_name))
    spath=os.path.join(result_save_path,result_dir_name)
    
# %% Chargement du moteur physique et des paramètres utilisé pour l'opti a partir du jupyter

    MoteurPhysique=MPHI(called_from_opti=True)
    if not type_grad==None:
        print("Chargement des fonctions du moteur physique...")
        MoteurPhysique.Effort_function=dill.load(open('../Simulation/function_moteur_physique'+type_grad,'rb'))
        MoteurPhysique.Theta = MoteurPhysique.Effort_function[-1]
        opti_variables_keys=MoteurPhysique.Theta
        print("Done")
    else:
        print("Attention aux chargement des params/fonctions du moteur physique")
        opti_variables_keys=['alpha0',
                             'cd1sa',
                              'cl1sa',
                              'cd0sa',
                              'coeff_drag_shift',
                              'coeff_lift_shift',
                              'coeff_lift_gain']
    
    opti_variables_keys.sort()        # Trie de la liste par ordre alphabétique. 
    "### Initialisation des dictionnaires de paramètres en fonctions des paramètres pour l'otpimisation." 
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
    
    "### Params PID gradient"
    gradient_kp=1
    gradient_kd=0
    gradient_ki=0
    G_sum =0
    G =0
    
    "####### Params pour l'opti ######"
    n_params_dropout = 0      # Bloque un certain nombre de paramètres à chaque térations pour augmenter le facteur aléatoire, si il est =0 on fait une identification paramètres par paramètres.
   
# %% Préparation des données pour l'opti
    print("Préparation des données...")

    "########################### params "
    if log_real==None:
        temp_df=raw_data.drop(columns=['alpha'])
        temp_df=temp_df.drop(columns=[i for i in temp_df.keys() if 'omegadot' in i])
    else:
        temp_df=raw_data.drop(columns=[i for i in raw_data.keys() if 'omegadot' in i])
    
    
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
    
    "### Préparation des données non randomizé pour comparaison en simulation. "
    X_test_sim = data_prepared.reset_index()[[i for i in data_prepared.keys() if not (('forces' in i) or ('torque' in i))]].values
    Y_test_sim = data_prepared.reset_index()[[i for i in data_prepared.keys() if (('forces' in i) or ('torque' in i))]].values
    print("Done")

    
# %% Fonction utilisé pour l'opti et la prépations des données
    "########################### funcs"
    "#####  Transformation d'un dictionnaire en une liste trié des données utiles pour l'identification. "
    def Dict_variables_to_X(Dict,opti_variables_keys=opti_variables_keys):
        V=[j  for key in np.sort([i for i in opti_variables_keys]) for j in np.array(Dict[key]).flatten()]
        return np.array(V)    
    
    "#####  Transformation d'un liste en un dictionnaire utisable par le moteur physique.   "
    def X_to_Dict_Variables(V, opti_variables_keys=opti_variables_keys, start_Dict_variables=start_Dict_variables):
        Dict={}
        counter=0
        "#### Ajout des valeurs utiliser pour l'identifation (paramètres de la liste)"
        for i in np.sort([i for i in opti_variables_keys]):
            L=len(np.array(start_Dict_variables[i]).flatten())
            S=np.array(start_Dict_variables[i]).shape
            Dict[i]=V[counter:counter+L].reshape(S)
            counter=counter+L
        "#### Ajout des autres paramètres du dictionnaire (non présent dans la liste, on utilise les valeurs de départ pour combler)"
        for i in start_Dict_variables.keys():
            if i not in opti_variables_keys:
                Dict[i]=start_Dict_variables[i]
        return Dict
    
    def generate_random_params(X_params,amp_dev=0.0,verbose=True):
        new_X_params=X_params*(1+amp_dev*(np.random.random(size=len(X_params))-0.5))    
        for p, params in enumerate(new_X_params):
            if params==0:
                new_X_params[p]=0.1
    
        print('\n[Randomization] X Old / X New variables: ')
        for i,j in zip(X_params,new_X_params):
            print(i,j,"diff = %f %%"%(round(abs(i-j)/i*100.0,2)))
    
        return new_X_params
    
    def model(X_params, x_data, y_data=None, compute_cost=None):
        "### Cette fonction permet de faire tourner le moteur physique pour un jeu de paramètres, avec un jeu de données d'entrée"
        
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
        MoteurPhysique.y_data=y_data
        
        if type(X_params)==dict:
            MoteurPhysique.Dict_variables=X_params
        else:
            MoteurPhysique.Dict_variables=X_to_Dict_Variables(X_params)
    
        if compute_cost==True:
            output, RMS_forces, RMS_torque =MoteurPhysique.compute_cost(joystick_input,t)
            return output, RMS_forces, RMS_torque
        else: 
            MoteurPhysique.compute_dynamics(joystick_input,t)
            d=np.r_[MoteurPhysique.forces,MoteurPhysique.torque]
        
            output=d.reshape((1,6))
            return output
    
    def cost(X_params,x_data,y_data,verbose=False, RMS=None):
        " ### Calcul de la fonction de cout : Moyenne de la sommes des erreurs au carré"
        "### Cette fonction peut renvoyer la RMS des forces et des couples si l'argument RMS est précisé. "
        if type(X_params)==dict:
            DictVariable_X = X_params
        else:
            DictVariable_X=X_to_Dict_Variables(X_params) 
    
        MoteurPhysique.Dict_variables=DictVariable_X
    
        used_x_batch=x_data
        used_y_batch=y_data
        
        C,RMS_forces, RMS_torque=0,0,0
        for k in range(len(used_x_batch)):
            cout, forces, torque = model(X_params, used_x_batch[k], used_y_batch[k], compute_cost=True) 
            RMS_forces+=forces
            RMS_torque+=torque
            C+=cout
    
        C/=len(used_x_batch)
        RMS_forces/=len(used_x_batch)
        RMS_torque/=len(used_x_batch)       # C=cout_forces*np.sum(sum_error_forces) + cout_torque*np.sum(sum_error_torque)
    
        if RMS==True:
            return C, RMS_forces , RMS_torque
        else:
            return C
    
    def compute_symbolic_gradient(X_params,x_data, y_data, W):
        " ### Calul du gradient numérique par le calcul symolique"
        "# MoteurPhysique.Dict_variables=X_to_Dict_Variables(X_params)"
        MoteurPhysique.Theta = opti_variables_keys
        Grad_list = []
        for i in range(len(x_data)):
            "Prepartion des entrées du systèmes"
            t,takeoff=x_data[i,0],x_data[i,1]
            speed=x_data[i,5:8]
            omega=x_data[i,11:14]
            q=x_data[i,14:18]
            joystick_input=x_data[i,18:]
    
            "Preparation des conditions initales pour le moteur physique"
            MoteurPhysique.speed=speed
            MoteurPhysique.q=q
            MoteurPhysique.omega=omega
            MoteurPhysique.R=tf3d.quaternions.quat2mat(MoteurPhysique.q).reshape((3,3))
            MoteurPhysique.takeoff=takeoff
            MoteurPhysique.y_data=y_data[i]
            " Calcule du gradient symbolique à partir de l'équation sympy"
            MoteurPhysique.compute_dynamics(joystick_input, t , compute_gradF=True)  
            Grad_list.append(MoteurPhysique.grad_cout)
    
        "Calul de la moyenne du gradient pour le batch"
        grad = sum(Grad_list[i] for i in range(len(Grad_list)))/ len(Grad_list) 
        grad=grad[0]
        if not np.linalg.norm(grad)==0:
            return grad
        else:
            return grad
    
# %% Découpage des données pour l'entrainement et le test

    # list_learning_rate=[learning_rate, learning_rate*10000, learning_rate*200, learning_rate*400, learning_rate*80, learning_rate*10]
    name=("Opti avec Learning rate initial = "+str(format(learning_rate, '.1E'))+" Avec PID = "+str(gradient_kp)+"/" +str(gradient_ki) + "/" + str(gradient_kd)+" avec des batch de taille :"+ str(train_batch_size))
    
    "########## Initialisation des paramètres #########"
    x_train = X_train
    y_train = Y_train
    
    begin = np.random.randint(len(X_train-100))
    end = begin+100
    x_train_cost = X_train[begin:end]
    y_train_cost = Y_train[begin:end]
    
    x_test = X_test
    y_test = Y_test
    # print(x_train)
    
# %% Création de la matrice des max des forces et couples
    max_forces=[max(abs(y_train[i][j]) for i in range(len(y_train))) for j in range(3)]
    max_torque=[max(abs(y_train[i][j+3]) for i in range(len(y_train))) for j in range(3)]
    W = np.diag([1/max_forces[0]**2, 1/max_forces[1]**2, 1/max_forces[2]**2, 1/max_torque[0]**2, 1/max_torque[1]**2, 1/max_torque[2]**2]) 
    MoteurPhysique.W = W
    
#%% Initatilisation des variables et du train score
    if log_real==True:
        current_train_score=1.
        start_Dict_variables = real_Dict_variables
        current_Dict_variables = start_Dict_variables
        best_Dict_variables = current_Dict_variables
    else:
        current_train_score=1.
        start_Dict_variables = X_to_Dict_Variables(generate_random_params\
                            (Dict_variables_to_X(real_Dict_variables),amp_dev=1,verbose=True))
        # start_Dict_variables["cd0sa"]=0.00999326172362672
        # start_Dict_variables["cd1sa"]=4.550179431280277
        # start_Dict_variables["cl1sa"]=4.999740931136897
        # start_Dict_variables["coeff_drag_shift"]=0.4999212041532855
        # start_Dict_variables["coeff_lift_gain"]=2.429539624079285
        # start_Dict_variables["coeff_lift_shift"]=0.06417931204686833
        # start_Dict_variables["wind_X"]=0.0
        # start_Dict_variables["wind_Y"]=0.0
    
        current_Dict_variables =start_Dict_variables
        best_Dict_variables = current_Dict_variables
    
# %% Création du monitor 
    print("Création du monitor...")
    monitor=OptiMonitor_MPL(name,opti_variables_keys=opti_variables_keys, params_real=true_params, opti_real=log_real, plot=plot)
    print("Done")
    monitor.x_data=data_prepared['t'].values     # Temps de la simulation pour la comparaison y_sim/y_real
    monitor.init_params=start_Dict_variables
    monitor.current_params=start_Dict_variables
    if log_real==False:
        print("Simulation avec les paramètres initiaux...")
        monitor.y_sim = [model(current_Dict_variables, X_test_sim[i]) for i in range(len(X_test_sim))]
        print("Done")
        monitor.y_real= Y_test_sim        # Valeur réel des efforts 
    z=0 # Compteur pour l'affichage des simulations en fonctions des epochs 
    
# %% création des fichier de sauvegarde ##########
    print("Ecriture des fichiers de sauvegarde de l'opti...")
    if spath is not None:
        with open(os.path.join(spath,"Params_initiaux.csv"),'w') as f:
            sdict={}
            sdict['log']=log_path
            for keys in true_params.keys():
                sdict[keys]=true_params[keys]
    
            sdict['learning_rate']=learning_rate
            sdict['n_epochs']=n_epochs
            sdict['gradient_kp']=gradient_kp
            sdict['gradient_kd']=gradient_kd
            sdict['gradient_ki']=gradient_ki
            sdict['train_batch_size']=train_batch_size
            sdict['n_params_dropout']=n_params_dropout        
            sdict['fitting_strategy']=fitting_strategy
            
            for i in sdict.keys():
                if type(sdict[i])==list:
                    sdict[i]=np.array(sdict[i]) 
                    
                if type(sdict[i])==np.ndarray:
                    sdict[i]=sdict[i].tolist() 
            spamwriter = csv.writer(f)
            spamwriter.writerow(sdict.keys())
            spamwriter.writerow(sdict.values())
    
            
        with open(os.path.join(spath, "Otpi_used.py"), 'w') as f:
            f.write(open(name_script).read())
            
            
    with open(os.path.join(spath,"results_opti.csv"),'w',newline='') as f:
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
           spamwriter = csv.writer(f)
           spamwriter.writerow(sdict.keys())
           
    with open(os.path.join(spath,"results.csv"),'w',newline='') as f:
        sdict={}
        for i in opti_variables_keys:
            sdict[i]=current_Dict_variables[i]
            
        sdict['train_score']=current_train_score
        
        for i in sdict.keys():
            if type(sdict[i])==list:
                sdict[i]=np.array(sdict[i]) 
                
            if type(sdict[i])==np.ndarray:
                sdict[i]=sdict[i].tolist() 
        spamwriter = csv.writer(f)
        spamwriter.writerow(sdict.keys())
    print("done \n Début de l'optimisation :")

# %% Début de l'optimization ####
    for i in range(n_epochs):
        "saving"
        
        if spath is not None:
            with open(os.path.join(spath,"results_opti.csv"),'a',newline='') as f:
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
                spamwriter = csv.writer(f)
                spamwriter.writerow(sdict.values())
               
        "opti loop"
        x_train_batch=[]
        y_train_batch=[]           

        "MAj du monitor des simulations." 
        if log_real==False:
            monitor.t = current_epoch
            current_test_score, monitor.RMS_forces, monitor.RMS_torque =cost(current_Dict_variables, x_test, y_test, verbose=True, RMS=True)
            monitor.y_eval, monitor.y_train = current_test_score,current_train_score
            monitor.update(epoch=True)
            n_update_sim=(n_epochs)/3
            if (monitor.t+1) % n_update_sim==0:
                monitor.update_sim_monitor(n_epoch=z)
                z+=1
            elif monitor.t==0:
                monitor.update_sim_monitor(n_epoch=z)
                z+=1
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
                
                "Descente du gradient de scipy (non utilisé dans le script)"
                if fitting_strategy=="scipy":
                    scaler=Dict_variables_to_X(start_Dict_variables)
                    scaler=np.array([i if i!=0 else 1.0 for i in scaler ])
                    X0_params=Dict_variables_to_X(current_Dict_variables)
    
                    res = minimize(cost(X0_params, x_train_batch, y_train_batch),method='SLSQP',
                          x0=X0_params,options={'maxiter': 100})
    
                    current_Dict_variables=X_to_Dict_Variables(res['x']*scaler)
                    
                "Descente du gradient customisé (utilisé ici)"
                if fitting_strategy=="custom_gradient":
                    "PID pour la descente du gradient"
                    X0_params=Dict_variables_to_X(current_Dict_variables)
                    G_pred = G
                    G=compute_symbolic_gradient(X0_params,x_train_batch, y_train_batch, W)
                    if type(G_pred)==int:
                        G_dot=G*0
                    else:
                        G_dot= (G_pred-G)
                    G_sum+=G
                    G_total = (gradient_kp * G) + (gradient_ki * G_sum) + (gradient_kd * G_dot)
                    
                    "Ajout d'aléatoire dans la descente"
                    # if not n_params_dropout==0:
                    #     for lzar in range(n_params_dropout):
                    #         kir=np.random.randint(0,len(X0_params))
                    #         G_total[kir]= 0
                        
                    "Descente : X = X0 - (Gain * Grad_norm * X_inital)"
                    new_X=X0_params  - (learning_rate*G_total* Dict_variables_to_X(start_Dict_variables))
                    # for h,o in enumerate(new_X):
                    #     if o<0:
                    #         if not type_grad in ["wind_only","params_plus_wind"]:
                    #             new_X[h]=0
                    #         else:
                    #             index_ = [opti_variables_keys.index(name) for name in ["wind_X", "wind_Y"]]
                    #             if not h in index_:
                    #                 new_X[h]=0
                                    
                    current_Dict_variables=X_to_Dict_Variables(new_X)
                print('########################')
                x_train_batch=[]
                y_train_batch=[]   
    
                current_train_score=cost(current_Dict_variables, x_train_cost, y_train_cost, verbose=True)
                print("epoch" ,current_epoch, "\n sample_nmbr = " , sample_nmbr,"/",len(x_train), "\n current_train_score = ", current_train_score)
                monitor.current_params=current_Dict_variables
                monitor.sample=sample_nmbr
                monitor.update()
    
                with open(os.path.join(spath,"results.csv"),'a',newline='') as f:
                   sdict={}
                   for i in opti_variables_keys:
                       sdict[i]=current_Dict_variables[i]
                       
                   sdict['train_score']=current_train_score
                   
                   for i in sdict.keys():
                       if type(sdict[i])==list:
                           sdict[i]=np.array(sdict[i]) 
                           
                       if type(sdict[i])==np.ndarray:
                           sdict[i]=sdict[i].tolist() 
                   spamwriter = csv.writer(f)
                   spamwriter.writerow(sdict.values())
                
        if x_test is not None and y_test is not None:
            new_test_score, monitor.RMS_forces, monitor.RMS_torque =cost(current_Dict_variables, x_test, y_test, verbose=True, RMS=True)
            if new_test_score<current_test_score:
                best_Dict_variables=current_Dict_variables
                with open(os.path.join(spath,"best_results.csv"),'w',newline='') as f:
                   sdict={}
                   for i in opti_variables_keys:
                       sdict[i]=best_Dict_variables[i]
                       
                   sdict['train_score']=current_train_score
                   
                   for i in sdict.keys():
                       if type(sdict[i])==list:
                           sdict[i]=np.array(sdict[i]) 
                           
                       if type(sdict[i])==np.ndarray:
                           sdict[i]=sdict[i].tolist() 
                   spamwriter = csv.writer(f)
                   spamwriter.writerow(sdict.keys())
                   
            current_test_score=new_test_score
            
            # for keys in monitor.dict_params_finish.keys():
            #     list_index = list_index + [opti_variables_keys.index(keys)]
                
        if log_real==False:
            monitor.y_sim = [model(current_Dict_variables, X_test_sim[i]) for i in range(len(X_test_sim))]
            monitor.y_real=Y_test_sim
                    
    for i in opti_variables_keys:
      
        print('########################\n real/start/current '+i+' :',
            real_Dict_variables[i],
            start_Dict_variables[i],
            best_Dict_variables[i])
        
        
        
# %% multirocessing
from multiprocessing import Pool

if __name__ == '__main__':
     
    log_real=[True, True]
    plot=[False,False]
    log_name=[""]
    train_batch_size=15
    learning_rate=[5e-2, 5e-2]
    fitting_strategy = ['custom_gradient']
    wind = [-2,-0.75]
    type_grad = ['params']
    
    x_r = [[log_real[1], plot[1],log_name[0],train_batch_size,learning_rate[j], fitting_strategy[0], wind, t] for j in range(1) for t in type_grad]
    L= len(x_r)
    for k in range(L):
        x_r[k].append(k+1)
        print('\n'+str(x_r[k]))
    r =input('Continue ? ( /n)')
    if r=="n":
        print("Stop proccess")
    else:
        pool = Pool(processes=2)
        pool.map(main_func, x_r)                         
        
        
        
