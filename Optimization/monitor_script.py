
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:41:02 2021

@author: mehdi
"""
import sys
sys.path.append('../')
import os
import matplotlib.pyplot as plt
from pylab import sort
import pandas as pd
import transforms3d as tf3d
import numpy as np
from Simulation.MoteurPhysique_class import MoteurPhysique as MPHI
import time
import dill as dill

if input('Log real ? (y/n)')=='y':    
    log_real=True
else:
    log_real=False

if log_real==True:
    opti_path_list=sort(os.listdir("../OptiResults/Opti_real/"))
    print(opti_path_list)
else:
    opti_path_list=sort(os.listdir("../OptiResults/Opti_sim/"))
    print(opti_path_list)
if input('Continue ? (/n)') =="n":
    print("Exit program")
else:
    trace =input('Tracer la/les simulation(s) des données avec le nouveau jeu de params ? ( /n)')
    Nb_opti =int(input('Nombre d opti a afficher : '))
    first_opti = int(input('Numéro de la première opti a afficher :'))
    MoteurPhysique=MPHI(called_from_opti=True)
    # MoteurPhysique.Effort_function=dill.load(open('../Simulation/function_moteur_physiquewind_only','rb'))

    for n_d in range(Nb_opti):
        if log_real==True:            
            opti_name =opti_path_list[n_d+first_opti]
            print(opti_name)
            
            opti_path_result = os.path.join("../OptiResults/Opti_real/"+opti_name )
            with open(opti_path_result+'/results.csv') as data:
                dic_results = pd.read_csv(data)
            plot_keys=dic_results.keys()
            opti_variables_keys= plot_keys.drop('train_score').to_list()
    
            with open(opti_path_result+'/Params_initiaux.csv') as init:
                dic_params_init = pd.read_csv(init)
                LR = dic_params_init['learning_rate'].values
                train_batch_size=dic_params_init['train_batch_size'].values
                name_plot = dic_params_init['log'].values[0].replace("../Logs/log_real/", 'Vol réel : ')
                name_plot=name_plot.replace('/log_real.csv', ' ')
        else:
            opti_name  = opti_path_list[n_d+first_opti]

            opti_path_result = os.path.join('../OptiResults/Opti_sim/'+opti_name )
            with open(opti_path_result+'/results.csv') as data:
                dic_results = pd.read_csv(data)
            plot_keys=dic_results.keys()
            opti_variables_keys= plot_keys.drop('train_score').to_list()
    
            with open(opti_path_result+'/Params_initiaux.csv') as init:
                dic_params_init = pd.read_csv(init)
                LR = dic_params_init['learning_rate'].values
                train_batch_size=dic_params_init['train_batch_size'].values
                name_plot = dic_params_init['log'].values[0].replace('../Logs/log_sim/', 'Vol simulé : ')
                name_plot=name_plot.replace('/log_real.csv', ' ')
                
        current_Dict_variables={}
        for k in MoteurPhysique.Dict_variables.keys():
            if k in dic_results.keys():
                column_index = dic_results.columns.get_loc(k)
                current_Dict_variables[k] =  dic_results.iloc[-1,column_index]
            else:
                column_index = dic_params_init.columns.get_loc(k)
                if type(dic_params_init.iloc[-1,column_index])==str:
                    current_Dict_variables[k]= pd.eval(dic_params_init.iloc[-1,column_index])
                else:
                    current_Dict_variables[k]= dic_params_init.iloc[-1,column_index]
                
        for keys in MoteurPhysique.Dict_variables.keys():
            if type(dic_params_init[keys].values[0])==str:
                MoteurPhysique.Dict_variables[keys]=pd.eval(dic_params_init[keys].values[0])
            else:
                MoteurPhysique.Dict_variables[keys]=dic_params_init[keys].values[0]
                                    

        fig = plt.figure("Resultat de l'opti : " +opti_name)
        list_fig=[]
        for i in range(len(plot_keys)):
            list_fig.append(fig.add_subplot(3,3,i+1))
             
        n=0
        fig.suptitle('optimisation pour le '+name_plot+ '\n avec LR = '+str(LR)+' et batch_size = '+str(train_batch_size))
        for keys, val in dic_results.items():
            list_fig[n].plot(val)
            list_fig[n].set_ylabel(keys)
            list_fig[n].set_xlabel('sample')
            if keys in opti_variables_keys:
                list_fig[n].axhline(dic_params_init[keys].values,color="black")
            if max(dic_results['train_score'].values)>50:
                if keys not in ['train_score', 'wind_X', 'wind_Y','eval_score']:
                    list_fig[n].set_ylim((0,dic_params_init[keys].values*50))
                elif keys in ["wind_X","wind_Y"]:
                    list_fig[n].set_ylim((-abs(dic_params_init[keys].values*3),abs(dic_params_init[keys].values*3)))
                elif keys=='train_score':
                    list_fig[n].set_ylim((0,val[0]*3))

            list_fig[n].grid()
            n+=1
            
        print('Logs : ' + dic_params_init['log'].values[0])
    
    #%% Simulation en fonction des paramètre

        raw_data=pd.read_csv(dic_params_init['log'].values[0])
    
        "########################### params "
        if log_real==False:
            temp_df=raw_data.drop(columns=['alpha'])
            temp_df=temp_df.drop(columns=[i for i in temp_df.keys() if 'omegadot' in i])
        else:
            temp_df=raw_data.drop(columns=[i for i in raw_data.keys() if 'omegadot' in i])
            temp_df=raw_data.drop(columns=['Unnamed: 0'])
    
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
    
        data_prepared=data_prepared.reset_index(drop=True)
        
        X_data=data_prepared[[i for i in data_prepared.keys() if not (('forces' in i) or ('torque' in i))]].values
        Y_data=data_prepared[[i for i in data_prepared.keys() if (('forces' in i) or ('torque' in i))]].values
    
        def Dict_variables_to_X(Dict,opti_variables_keys=opti_variables_keys):
            V=[j  for key in np.sort([i for i in opti_variables_keys]) for j in np.array(Dict[key]).flatten()]
            return np.array(V)    
        
            "#####  Transformation d'un liste en un dictionnaire utisable par le moteur physique.   "
        def X_to_Dict_Variables(V, opti_variables_keys=opti_variables_keys, start_Dict_variables=current_Dict_variables):
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
        
        def model(X_params,x_data, y_data=None, log_real=False, MoteurPhysique=MoteurPhysique):
            "### Cette fonction permet de faire tourner le moteur physique pour un jeu de paramètres, avec un jeu de données d'entrée"
            t,takeoff=x_data[0],x_data[1]
            if log_real==True:
                offset=3
            else:
                offset=0
            speed=x_data[5:8]
            omega=x_data[11+offset:14+offset]
            q=x_data[14+offset:18+offset]
            joystick_input=x_data[18+offset:]
            
            MoteurPhysique.speed=speed
            MoteurPhysique.q=q
            MoteurPhysique.omega=omega
            MoteurPhysique.R=tf3d.quaternions.quat2mat(MoteurPhysique.q)
            MoteurPhysique.takeoff=takeoff
            if not y_data is None:
                MoteurPhysique.y_data=y_data
            
            if type(X_params) is dict:
                MoteurPhysique.Dict_variables=X_params
            else:
                MoteurPhysique.Dict_variables=X_to_Dict_Variables(X_params)

            MoteurPhysique.compute_dynamics(joystick_input,t)
            d=np.r_[MoteurPhysique.forces,MoteurPhysique.torque]
            output=d.reshape((1,6))
            return output
        
        X_params=Dict_variables_to_X(current_Dict_variables)
        if log_real==True:
            X_params_init=Dict_variables_to_X(dic_params_init)
        else:
            dict_params_init={}
            for keys, values in dic_results.items():
                if keys not in ['train score']:
                    dict_params_init[keys]=values[1]
            X_params_init=Dict_variables_to_X(dict_params_init)

        if not trace=='n':
            if log_real==True:
                beg=(int(np.random.random()*len(X_data))-10000)*0
                if beg>len(X_data)-10000:
                    beg=len(X_data)-10100
                elif beg<0:
                    beg=10
                en=len(X_data)
            else:
                beg=0
                en=len(X_data)

            t1 = time.time() 
            y_sim_end = [model(X_params, X_data[beg+v], log_real=log_real) for v in range(en-beg)]
            t2 = time.time() 
            print("End for sim with end params in : "+str(t2-t1)+" s")
            y_sim_begin=[model(X_params_init, X_data[beg+v], log_real=log_real) for v in range(en-beg)]
            t3 = time.time()    
            print("End for sim with begin params in : "+str(t3-t2)+" s")
            
            liste_name=['Force', 'Couple']
            fig_sim=plt.figure("Simulation neg pour l'opti "+opti_name)
            
            L_real=['black','black','black'] 
            L_sim_end=['red','red','red']
            L_sim_begin=['green','green','green']
            for w in range(6):
                
                x_plot = [X_data[beg+j][0] for j in range(en-beg)]
                y_plot = [Y_data[beg+j][w] for j in range(en-beg)]
                
                y_end=[y_sim_end[j][0][w] for j in range(en-beg)]
                y_begin=[y_sim_begin[j][0][w] for j in range(en-beg)]

                figure=fig_sim.add_subplot(2,3,w+1)
              
                if w<3:
                    figure.plot(x_plot,y_plot, label="real data", color=L_real[w])
                    figure.plot(x_plot, y_begin, label="simu avant opti", color=L_sim_begin[w], linestyle='--')
                    figure.plot(x_plot, y_end, label="simu post opti", color=L_sim_end[w],linestyle='--')
                    figure.set_ylabel(liste_name[0]+"["+str(w)+"] (N)")
                else:
                    figure.plot(x_plot,y_plot, label="real data", color=L_real[w-3])
                    figure.plot(x_plot, y_begin, label="simu avant opti", color=L_sim_begin[w-3],linestyle='--')
                    figure.plot(x_plot, y_end, label="simu post opti", color=L_sim_end[w-3],linestyle='--')
                    figure.set_ylabel(liste_name[1]+"["+str(w-3)+"] (N/m)")
                figure.legend()
                figure.set_xlabel('Time (s)')
                figure.grid()