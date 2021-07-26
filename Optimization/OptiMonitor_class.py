#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:16:33 2021

@author: l3x
"""


import numpy as np
import time
import matplotlib.pyplot as plt
from pylab import * 
import os 
import json

class OptiMonitor_MPL():
    def __init__(self,name=None, opti_variables_keys=None, params_real=None):
        
        self.name=name if name!=None else str(int(time.time()))
        
        self.fig = plt.figure(self.name)
        self.fig.patch.set_facecolor('lightgrey') 
        self.ax01= self.fig.add_subplot(3,2,1)      
        self.ax02= self.fig.add_subplot(3,2,3)
        self.ax03= self.fig.add_subplot(3,2,5)

        self.ax1= self.fig.add_subplot(4,4,3)
        self.ax2= self.fig.add_subplot(4,4,4)
        
        self.ax3= self.fig.add_subplot(4,4,7)
        self.ax4= self.fig.add_subplot(4,4,8)
        self.ax5= self.fig.add_subplot(4,4,11)
        self.ax6= self.fig.add_subplot(4,4,12)
        self.ax7= self.fig.add_subplot(4,4,15)
        self.ax8= self.fig.add_subplot(4,4,16)

        self.ax_dic = [self.ax01, self.ax02, self.ax03,self.ax1,self.ax2,self.ax3, self.ax5,self.ax7, self.ax4, self.ax6,self.ax8]
        
        self.fig_sim = plt.figure(self.name+" evolution du fit des simulation en fonction de l'epoch") 
        self.fig_sim.patch.set_facecolor('lightgrey') 

        # Liste des couleurs pour les différentes courbes des erreurs 
        self.color_list = ['b','g','r','c','m','y','k', '0.70', '0.30', (0.2,0.9,0.1)]
        if opti_variables_keys==None:
            self.opti_variables_keys=['alpha_stall',
                                      'largeur_stall',
                                      'cd1sa',
                                      'cl1sa',
                                      'cd0sa',
                                      'cd1fp',
                                      'cd0fp',
                                      'coeff_drag_shift',
                                      'coeff_lift_shift',
                                      'coeff_lift_gain']
        else:
            self.opti_variables_keys=opti_variables_keys
            
        self.opti_variables_keys.sort()
        self.t0=time.time()
        self.t,self.y_train,self.y_eval=0,0,0
        self.t_data=[self.t]
        self.train_data=[self.y_train]
        self.eval_data=[self.y_eval]  
        self.RMS_forces_list = [0]
        self.RMS_torque_list = [0]
        self.L_real=['darkgreen','darkred','darkblue'] 
        self.L_sim=['lightgreen','tomato','skyblue']
        self.train_score_curve, = self.ax01.plot(self.train_data,
                                                label="train",
                                                marker="x")
        self.eval_score_curve, = self.ax01.plot(self.eval_data,
                                              label="eval",
                                              marker="o")
        # Init des différents graphiques
        for i in [self.ax1, self.ax2]:
            i.grid()
            i.set_xlabel('data sample')
            i.set_ylabel('error %')
            i.legend()
            
        name_graph =['Force(N)', 'Torque(N/m)']
        for p, i in enumerate([self.ax3, self.ax4]):
            i.grid()
            i.set_xlabel('time')
            i.set_ylabel(name_graph[p])
            i.legend()   

        for p, i in enumerate([self.ax02, self.ax03]):
            i.grid()
            i.set_xlabel('time')
            i.set_ylabel("Percent of cost for "+name_graph[p])
            i.legend()
            
        # Chargement des paramètres si il est pas fait lors de l'initialisation de la classe (par défaut prend le dernier log)
        if params_real==None:
           log_dir_path=os.path.join("../Logs/",os.listdir("../Logs/")[-1])
           params_real_path =os.path.join(log_dir_path,"params.json")
           with open(params_real_path,"r") as f:
            self.params_real=json.load(f)
        else:
            self.params_real=params_real
        self.title=self.name+ " pour un vol en mode "+ str(self.params_real["mode"])
       
    
        self.error_dic = {}
        self.sample_percent=[]
        self.sample=0
        self.dict_params_finish={}
        
        self.fig.suptitle(self.title + " : Evolution du fitting en fonction des epochs")

        
    def legend(self, epoch=None, dic_fig_sim=None):
        # Cette fonction permet de legender et commenter les figures par groupe. 
        if epoch==False:
            for i in [self.ax1, self.ax2]:
                i.set_xlabel('data sample (%)')
                i.set_ylabel('error (%)')
                i.legend()  
                
        elif dic_fig_sim:
            name_graph =['Force(N)', 'Torque(N/m)']
            for p, i in enumerate(dic_fig_sim):
                i.grid()
                i.set_xlabel('time')
                if (p+1)%2==0:
                    i.set_ylabel(name_graph[1])
                else:
                    i.set_ylabel(name_graph[0])
                i.legend()   
        else: 
            name_graph =['Force(N)', 'Torque(N/m)']
            for p, i in enumerate(self.ax_dic[5:]):
                i.set_xlabel('time')
                if p>2:
                    i.set_ylabel(name_graph[1])
                else:
                    i.set_ylabel(name_graph[0])
                i.legend()   
    
            for p, i in enumerate([self.ax02, self.ax03]):
                i.set_xlabel('epoch')
                i.set_ylabel("Percent of cost ")
                i.legend()  
                
            self.ax01.set_xlabel('epoch')
            self.ax01.set_ylabel('Cost')
            self.ax01.legend()
            
        return 
    
    def update(self, epoch=None):
        # Cette fonction permet de mettre à jour le moniteur, il permet de mettre à jour les graphes correspondant. 
        current_params = self.current_params

        if  not epoch==True:
            # Cela met à jour les graphs des erreurs à chaque batch
            for ax in [self.ax1, self.ax2]:
                ax.clear()
                ax.grid()
            self.sample_percent = self.sample_percent + [self.t + self.sample/len(self.x_data)]
            for p, keys in enumerate(self.opti_variables_keys):
                error = (current_params[keys] - self.params_real[keys]) / self.params_real[keys] * 100
                if keys in self.error_dic:
                    if keys in self.dict_params_finish:
                        self.error_dic.update({keys :self.error_dic[keys]+ [self.dict_params_finish[keys]]})
                    else:
                        self.error_dic.update({keys : self.error_dic[keys] + [error]})
                else :
                    self.error_dic.update({keys:[error]})
                if int(p<len(self.opti_variables_keys)/2):
                    self.ax_dic[3].plot(self.sample_percent, self.error_dic[keys], label=keys+" = "+str(np.round(current_params[keys],4)), color=self.color_list[p])
                    self.ax_dic[3].set_ylim(-100,100)
                else:
                    self.ax_dic[4].plot(self.sample_percent, self.error_dic[keys], label=keys+" = "+str(np.round(current_params[keys],4)), color=self.color_list[p])
                    self.ax_dic[4].set_ylim(-100,100)
            self.legend(epoch=False)
            
        else:
            ##### Met à jour les graphs à chaque epochs. 
            # ### Verifiation si la valeur finale est atteinte. 
            # if self.t>0:
            #     for keys_params in self.error_dic.keys():
            #         if len(self.dict_params_finish)>=4:
            #             if keys_params in ['coeff_lift_gain', 'coeff_lift_shift']:
            #                 begin = int((self.t-1)*len(self.error_dic[keys_params])/self.t)
            #                 first_mean = np.mean(self.error_dic[keys_params][begin: begin+ 2*int(len(self.error_dic[keys_params][begin:-1])/3)])
            #                 second_mean = np.mean(self.error_dic[keys_params][begin+2*int(len(self.error_dic[keys_params][begin:-1])/3):-1])
            #                 if abs(first_mean-second_mean)<0.05:
            #                     self.dict_params_finish[keys_params]=(first_mean+second_mean)/2
            #                     print("end opti for :" + keys_params + " with mean = " +str( self.dict_params_finish[keys_params]))
            #         else: 
            #             if keys_params not in self.dict_params_finish.keys():
            #                 if keys in not ['coeff_lift_gain', 'coeff_lift_shift']:
            #                     begin = int((self.t-1)*len(self.error_dic[keys_params])/self.t)
            #                     first_mean = np.mean(self.error_dic[keys_params][begin: begin+ 2*int(len(self.error_dic[keys_params][begin:-1])/3)])
            #                     second_mean = np.mean(self.error_dic[keys_params][begin+2*int(len(self.error_dic[keys_params][begin:-1])/3):-1])
            #                     if abs(first_mean-second_mean)<0.05 :
            #                         if keys_params not in self.dict_params_finish.keys():
            #                             self.dict_params_finish[keys_params]=(first_mean+second_mean)/2
            #                             print("end opti for :" + keys_params + " with mean = " +str( self.dict_params_finish[keys_params]))
            
            # print("End opti for :", self.dict_params_finish.keys())
            for ax in [self.ax01,self.ax02,self.ax03,self.ax3, self.ax4, self.ax5, self.ax6, self.ax7,self.ax8]:
                ax.clear()
                ax.grid()

            ### Ajout des nouvelles données
            self.t_data=self.t_data+[self.t]
            self.train_data=self.train_data+[self.y_train]
            self.eval_data=self.eval_data+[self.y_eval]
            
            self.train_score_curve.set_data(self.t_data,self.train_data)
            self.eval_score_curve.set_data(self.t_data,self.eval_data)
            
            ### MAJ Affichage du cout par epoch 
            self.train_score_curve, = self.ax01.plot(self.train_data,
                                                   label="train",
                                                   marker="x")
            self.eval_score_curve, = self.ax01.plot(self.eval_data,
                                                  label="eval",
                                                  marker="o")
            
            ### MAJ de RMS error des efforts 
            self.RMS_forces_list = self.RMS_forces_list +[self.RMS_forces/self.y_eval]
            self.RMS_torque_list = self.RMS_torque_list +[self.RMS_torque/self.y_eval]
     
            self.ax_dic[1].plot(self.t_data, self.RMS_forces_list, label='RMS forces error',linestyle='--')
            self.ax_dic[2].plot(self.t_data, self.RMS_torque_list, label='RMS torque error',linestyle='--')
                
    
            #### MAJ des courbes des simulation des efforts. 
            for i in range(3):
                self.ax_dic[i+5].plot(self.x_data, [self.y_sim[j][0][i] for j in range(len(self.y_sim))], label="Force_sim["+str(i)+"]",color=self.L_sim[i])
                self.ax_dic[i+5].plot(self.x_data, self.y_real[:,i], label="Force_real["+str(i)+"]",color=self.L_real[i],linestyle='--' )
                self.ax_dic[i+8].plot(self.x_data, [self.y_sim[j][0][3+i] for j in range(len(self.y_sim))], label="Torque_sim["+str(i)+"]", color=self.L_sim[i])
                self.ax_dic[i+8].plot(self.x_data, self.y_real[:,3+i], label="Torque_real["+str(i)+"]",color=self.L_real[i],linestyle='--' )
                
            self.legend(epoch=True)

                
        self.fig.suptitle(self.title)
        plt.pause(0.001)

        return
    
    def update_sim_monitor(self, n_epoch=None):
        #MAJ de la deuxième fenêtre, évolution de la simulation en fonctions des paramètres. 
        self.ax_sim1 = self.fig_sim.add_subplot(4,6,(6*n_epoch)+1) 
        self.ax_sim2 = self.fig_sim.add_subplot(4,6,(6*n_epoch)+3)  
        self.ax_sim3 = self.fig_sim.add_subplot(4,6,(6*n_epoch)+5) 
        
        self.ax_sim4 = self.fig_sim.add_subplot(4,6,(6*n_epoch)+2) 
        self.ax_sim5 = self.fig_sim.add_subplot(4,6,(6*n_epoch)+4)  
        self.ax_sim6 = self.fig_sim.add_subplot(4,6,(6*n_epoch)+6)
        dic_fig_sim=[self.ax_sim1,self.ax_sim4,self.ax_sim2,self.ax_sim5,self.ax_sim3,self.ax_sim6]
        for i in range(3):
           dic_fig_sim[2*i].plot(self.x_data, [self.y_sim[j][0][i] for j in range(len(self.y_sim))], label="Force_sim["+str(i)+"]",color=self.L_sim[i])
           dic_fig_sim[2*i].plot(self.x_data, self.y_real[:,i], label="Force_real["+str(i)+"]",color=self.L_real[i],linestyle='--' )
           dic_fig_sim[2*i+1].plot(self.x_data, [self.y_sim[j][0][3+i] for j in range(len(self.y_sim))], label="Torque_sim["+str(i)+"]", color=self.L_sim[i])
           dic_fig_sim[2*i+1].plot(self.x_data, self.y_real[:,3+i], label="Torque_real["+str(i)+"]",color=self.L_real[i],linestyle='--' )
        self.legend(dic_fig_sim=dic_fig_sim)
        plt.pause(0.001)
        return

    
    def launch(self):
        plt.ion()
        while 1:
            time.sleep(0.05)
            self.update()

# o=OptiMonitor_MPL()
# o.launch()
# o.update()
