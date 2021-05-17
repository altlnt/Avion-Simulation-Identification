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
    def __init__(self,name=None, opti_variables_keys=None):
        
        self.name=name if name!=None else str(int(time.time()))
        
        self.fig = plt.figure()
        
        self.ax= self.fig.add_subplot(121)
        self.ax1= self.fig.add_subplot(5,4,3)
        self.ax2= self.fig.add_subplot(5,4,4)
        self.ax3= self.fig.add_subplot(5,4,7)
        self.ax4= self.fig.add_subplot(5,4,8)
        self.ax5= self.fig.add_subplot(5,4,11)
        self.ax6= self.fig.add_subplot(5,4,12)        
        self.ax7= self.fig.add_subplot(5,4,15)
        self.ax8= self.fig.add_subplot(5,4,16)
        self.ax9= self.fig.add_subplot(5,4,19)
        self.ax10= self.fig.add_subplot(5,4,20)

        self.ax_dic = [self.ax,self.ax1,self.ax2,self.ax3,self.ax4,self.ax5,self.ax6,self.ax7,self.ax8,self.ax9,self.ax10]
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
                                      'coef_lift_gain']
        else:
            self.opti_variables_keys=opti_variables_keys()
            
        self.t0=time.time()
        self.t,self.y_train,self.y_eval=0,0,0
        self.t_data=[self.t]
        self.train_data=[self.y_train]
        self.eval_data=[self.y_eval]     

       
        # self.train_score_scat, = self.ax.scatter(self.t_data,self.train_data,label="train")
        # self.eval_score_scat, = self.ax.scatter(self.t_data,self.eval_data,label="eval")
        self.train_score_curve, = self.ax.plot(self.train_data,
                                               label="train",
                                               marker="x")
        self.eval_score_curve, = self.ax.plot(self.eval_data,
                                              label="eval",
                                              marker="o")
        for i in self.ax_dic:
            i.grid()
            i.set_xlabel('time')
            i.set_ylabel('error %')
            i.legend()

        # self.datadir=sort(os.listdir(os.path.join(os.getcwd(),"../Logs")))[-1]
        path = sort(os.listdir(os.getcwd()+"/../Logs"))[-1] + "/params.json"
        self.params_real = json.load(open(os.path.join(os.getcwd()+"/../Logs",path)))

        self.current_params = {}
        self.init_params ={}
    
        self.error_dic = {}
    def update(self):
        current_params = self.current_params
        init_params = self.init_params
        self.ax.clear()
        self.train_score_curve, = self.ax.plot(self.train_data,
                                               label="train",
                                               marker="x")
        self.eval_score_curve, = self.ax.plot(self.eval_data,
                                              label="eval",
                                              marker="o")
        # self.t=time.time()-self.t0
        # self.y=10*np.exp(-0.01*self.t)
        
        # self.y_train,self.y_eval=self.y,self.y+1
        
        self.t_data=self.t_data+[self.t]
        self.train_data=self.train_data+[self.y_train]
        self.eval_data=self.eval_data+[self.y_eval]
        
        self.train_score_curve.set_data(self.t_data,self.train_data)
        self.eval_score_curve.set_data(self.t_data,self.eval_data)

        for p, keys in enumerate(self.opti_variables_keys):
            error = abs(current_params[keys] - self.params_real[keys]) / self.params_real[keys] * 100
            if keys in self.error_dic:
                self.error_dic.update({keys : self.error_dic[keys] + [error]})
            else :
                self.error_dic.update({keys:[error]})
            self.ax_dic[p+1].clear()
            self.ax_dic[p+1].plot(self.error_dic[keys], label=keys+" = "+str(current_params[keys]), marker="x")
            self.ax_dic[p+1].grid()
            self.ax_dic[p+1].legend()
            self.ax_dic[p+1].relim()
            self.ax_dic[p+1].autoscale_view()
        # for keys,values in self.params_real.items():
        #     values = self.dict_to_str(self.params_real, keys)
        #     if keys in current_params:
        #         current_values = self.dict_to_str(current_params, keys)
        #     else:
        #         current_values ='None'
        #     if keys in init_params:
        #         init_values = self.dict_to_str(init_params, keys)
        #     else: 
        #         init_values = 'None'
        #     if current_values =='None' and init_values=='None':
        #         self.info = self.info+"\n"
        #     else:
        #         self.info=self.info+"\n"+keys +': real'+' :  '+values+' init : '+init_values+' : current'+' : '+current_values
         
        # plt.text(0.2,0.5, s=self.info,verticalalignment='center', horizontalalignment='center', fontsize=10, transform=self.ax.transAxes)

        # self.train_score_scat.set_data(self.t_data,self.train_data)
        # self.eval_score_scat.set_data(self.t_data,self.eval_data)
        self.ax.grid()
        self.ax.legend()

        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)

        return
    
    # def dict_to_str(self, dic, keys):
    #     values =''
    #     if type(dic[keys]) == list:
    #         if type(dic[keys][0])==int or type(dic[keys][0])==float:
    #             values = "["
    #             for j in range(len(dic[keys])-1):
    #                 values = values + str(dic[keys][j])+","
    #             values=values+str(dic[keys][-1])+"]"
                
    #         elif type(dic[keys])==float or type(dic[keys])==int:
    #                values = str(dic[keys])
                   
    #         else :
    #             if type(dic[keys][0][0])==float or type(dic[keys][0][0])==int:
    #                 if keys=='inertie':
    #                     values = "["+str(dic[keys][0][0])+","\
    #                         +str(dic[keys][1][1])+","+str(dic[keys][2][2])+"]"
                   
    #                 else : 
    #                     for i in range(len(dic[keys][0])-1):
    #                         values=str(dic[keys][i])+","
    #                     values="["+values+str(dic[keys][-1])+"]"
    #             else : 
    #                 values ="["+str(dic[keys][0][0])+","\
    #                         +str(dic[keys][0][1])+","+str(dic[keys][0][2])+"]"+"\n ["+str(dic[keys][1][0])+","\
    #                         +str(dic[keys][1][1])+","+str(dic[keys][1][2])+"]"+"\n ["+str(dic[keys][2][0])+","\
    #                         +str(dic[keys][2][1])+","+str(dic[keys][2][2])+"]"+"\n ["+str(dic[keys][3][0])+","\
    #                         +str(dic[keys][3][1])+","+str(dic[keys][3][2])+"]"+"\n ["+str(dic[keys][4][0])+","\
    #                         +str(dic[keys][4][1])+","+str(dic[keys][4][2])+"]"
               
                   
    #     elif type(dic[keys])==float or type(dic[keys])==int:
    #         values=str(dic[keys])

    #     else:

    #         if keys=='inertie':
    #             values = "["+str(dic[keys][0][0])+","\
    #                         +str(dic[keys][1][1])+","+str(dic[keys][2][2])+"]"
    #         elif keys=="cp_list" : 
    #             values ="["+str(dic[keys][0][0])+","\
    #                 +str(dic[keys][0][1])+","+str(dic[keys][0][2])+"]"+"\n ["+str(dic[keys][1][0])+","\
    #                 +str(dic[keys][1][1])+","+str(dic[keys][1][2])+"]"+"\n ["+str(dic[keys][2][0])+","\
    #                 +str(dic[keys][2][1])+","+str(dic[keys][2][2])+"]"+"\n ["+str(dic[keys][3][0])+","\
    #                 +str(dic[keys][3][1])+","+str(dic[keys][3][2])+"]"+"\n ["+str(dic[keys][4][0])+","\
    #                 +str(dic[keys][4][1])+","+str(dic[keys][4][2])+"]"
            
    #         elif type(dic[keys])==float or type(dic[keys])==int:
    #             values = str(dic[keys])
    #         else : 
    #             if np.size(dic[keys])==1:
    #                 values = str(dic[keys])
    #             else :
    #                 values = "["
    #                 for j in range(len(dic[keys])-1):
    #                     values = values + str(dic[keys][j])+","
    #                 values=values+str(dic[keys][-1])+"]"
    #     return values
    
    def launch(self):
        plt.ion()
        while 1:
            time.sleep(0.05)
            self.update()

# o=OptiMonitor_MPL()
# o.launch()
# o.update()
