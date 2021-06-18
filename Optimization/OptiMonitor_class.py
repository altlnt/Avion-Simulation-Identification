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
            i.set_ylabel("RMS error "+name_graph[p])
            i.legend()   
            
        if params_real==None:
           log_dir_path=os.path.join("../Logs/",os.listdir("../Logs/")[-1])
           params_real_path =os.path.join(log_dir_path,"params.json")
           with open(params_real_path,"r") as f:
            self.params_real=json.load(f)
        else:
            self.params_real=params_real
        self.title=self.name+ " pour un vol en mode "+ str(self.params_real["mode"])
        self.current_params = {}
        self.init_params ={}
    
        self.error_dic = {}
        self.sample_percent=[]
        self.sample=0
    def update(self, epoch=None):
        current_params = self.current_params

        if epoch==None:
            for ax in [self.ax1, self.ax2]:
                ax.clear()
                ax.grid()
            self.sample_percent = self.sample_percent + [self.t + self.sample/len(self.x_data)]
            for p, keys in enumerate(self.opti_variables_keys):
                error = abs(current_params[keys] - self.params_real[keys]) / self.params_real[keys] * 100
                if keys in self.error_dic:
                    self.error_dic.update({keys : self.error_dic[keys] + [error]})
                else :
                    self.error_dic.update({keys:[error]})
                if int(p<len(self.opti_variables_keys)/2):
                    self.ax_dic[3].plot(self.sample_percent, self.error_dic[keys], label=keys+" = "+str(np.round(current_params[keys],4)), color=self.color_list[p])
                    self.ax_dic[3].set_ylim(0,100)
                else:
                    self.ax_dic[4].plot(self.sample_percent, self.error_dic[keys], label=keys+" = "+str(np.round(current_params[keys],4)), color=self.color_list[p])
                    self.ax_dic[4].set_ylim(0,100)
            for ax in [self.ax1, self.ax2]:
                ax.legend()
                ax.relim()
            
        else: 
            
            for ax in [self.ax01,self.ax02,self.ax03,self.ax3, self.ax4, self.ax5, self.ax6, self.ax7,self.ax8]:
                ax.clear()
                ax.grid()
            #### MAJ affichage pour le pourcentage d'erreur
            self.t_data=self.t_data+[self.t]
            self.train_data=self.train_data+[self.y_train]
            self.eval_data=self.eval_data+[self.y_eval]
            
            self.train_score_curve.set_data(self.t_data,self.train_data)
            self.eval_score_curve.set_data(self.t_data,self.eval_data)
            
            ### Affichage du cout par epoch 
            self.train_score_curve, = self.ax01.plot(self.train_data,
                                                   label="train",
                                                   marker="x")
            self.eval_score_curve, = self.ax01.plot(self.eval_data,
                                                  label="eval",
                                                  marker="o")
            
            ### MAJ de RMS error des efforts 
            self.RMS_forces_list = self.RMS_forces_list +[self.RMS_forces]
            self.RMS_torque_list = self.RMS_torque_list +[self.RMS_torque]
     
            self.ax_dic[1].plot(self.t_data, self.RMS_forces_list, label='RMS forces error',linestyle='--')
            self.ax_dic[2].plot(self.t_data, self.RMS_torque_list, label='RMS torque error',linestyle='--')
                
    
            #### MAJ des courbes des simulation des efforts. 
            for i in range(3):
                self.ax_dic[i+5].plot(self.x_data, [self.y_sim[j][0][i] for j in range(len(self.y_sim))], label="Force_sim["+str(i)+"]",color=self.L_sim[i])
                self.ax_dic[i+5].plot(self.x_data, self.y_real[:,i], label="Force_real["+str(i)+"]",color=self.L_real[i],linestyle='--' )
                self.ax_dic[i+8].plot(self.x_data, [self.y_sim[j][0][3+i] for j in range(len(self.y_sim))], label="Torque_sim["+str(i)+"]", color=self.L_sim[i])
                self.ax_dic[i+8].plot(self.x_data, self.y_real[:,3+i], label="Torque_real["+str(i)+"]",color=self.L_real[i],linestyle='--' )
                
            for ax in [self.ax01,self.ax02,self.ax03,self.ax3, self.ax4, self.ax5, self.ax6, self.ax7,self.ax8]:
                ax.legend()
                ax.relim()
                
        self.fig.suptitle(self.title)
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
