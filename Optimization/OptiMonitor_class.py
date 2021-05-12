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
    def __init__(self,name=None):
        
        self.name=name if name!=None else str(int(time.time()))
        
        self.fig = plt.figure()
        
        self.ax= self.fig.add_subplot(111)
        
        


        
        self.t0=time.time()
        self.t,self.y_train,self.y_eval=0,0,0
        self.t_data=[self.t]
        self.train_data=[self.y_train]
        self.eval_data=[self.y_eval]     

        self.datadir=sort(os.listdir(os.path.join(os.getcwd(),"../Logs")))[-1]
        self.params_real = json.load(open(os.path.join(os.getcwd(),"../Logs",self.datadir,"params.json")))
        
        # self.train_score_scat, = self.ax.scatter(self.t_data,self.train_data,label="train")
        # self.eval_score_scat, = self.ax.scatter(self.t_data,self.eval_data,label="eval")
        self.train_score_curve, = self.ax.plot(self.train_data,
                                               label="train",
                                               marker="x")
        self.eval_score_curve, = self.ax.plot(self.eval_data,
                                              label="eval",
                                              marker="o")
        
        self.params_current = dict()
        self.ax.grid()
        self.ax.legend()

        self.info = ''
    def update(self, current_valeur):
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
        
        self.params_current = current_valeur
        self.info=''
        
        for keys,values in self.params_real.items():
            values = self.dict_to_str(self.params_real, keys)
            if keys in self.params_current:
                current_values = self.dict_to_str(self.params_current, keys)
                self.info=self.info+"\n"+keys+'_real'+' :  '+values+' et ' +keys+'_current'+' : '+current_values
            else :
                #self.info=self.info+"\n"+keys+'_real'+' : '+values 
                print('1')
        plt.text(0.0,0.0, s=self.info, fontsize=8, transform=self.ax.transAxes)

        # self.train_score_scat.set_data(self.t_data,self.train_data)
        # self.eval_score_scat.set_data(self.t_data,self.eval_data)
        self.ax.grid()
        self.ax.legend()

        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)

        return
    
    def dict_to_str(self, dic, keys):
        values=''
        if type(dic[keys]) == list:
            if type(dic[keys][0])==int or type(dic[keys][0])==float:
                values = "["
                for j in range(len(dic[keys])-1):
                    values = values + str(dic[keys][j])+","
                values=values+str(dic[keys][-1])+"]"
            else :
                if type(dic[keys][0][0])==float or type(dic[keys][0][0])==int:
                    if keys=='inertie':
                        values = "["+str(dic[keys][0][0])+","\
                            +str(dic[keys][1][1])+","+str(dic[keys][2][2])+"]"
                    elif keys=="cp_list" : 
                        values ="["+str(dic[keys][0][0])+","\
                            +str(dic[keys][0][1])+","+str(dic[keys][0][2])+"]"+"\n ["+str(dic[keys][1][0])+","\
                            +str(dic[keys][1][1])+","+str(dic[keys][1][2])+"]"+"\n ["+str(dic[keys][2][0])+","\
                            +str(dic[keys][2][1])+","+str(dic[keys][2][2])+"]"+"\n ["+str(dic[keys][3][0])+","\
                            +str(dic[keys][3][1])+","+str(dic[keys][3][2])+"]"+"\n ["+str(dic[keys][4][0])+","\
                            +str(dic[keys][4][1])+","+str(dic[keys][4][2])+"]"
                    else : 
                        for i in range(len(dic[keys][0])-1):
                            values=str(dic[keys][i])+","
                        values="["+values+str(dic[keys][-1])+"]"
               
        elif type(dic[keys])==float or type(dic[keys])==int:
            values=str(dic[keys])

        else:

            if keys=='inertie':
                values = "["+str(dic[keys][0][0])+","\
                            +str(dic[keys][1][1])+","+str(dic[keys][2][2])+"]"
            elif keys=="cp_list" : 
                values ="["+str(dic[keys][0][0])+","\
                    +str(dic[keys][0][1])+","+str(dic[keys][0][2])+"]"+"\n ["+str(dic[keys][1][0])+","\
                    +str(dic[keys][1][1])+","+str(dic[keys][1][2])+"]"+"\n ["+str(dic[keys][2][0])+","\
                    +str(dic[keys][2][1])+","+str(dic[keys][2][2])+"]"+"\n ["+str(dic[keys][3][0])+","\
                    +str(dic[keys][3][1])+","+str(dic[keys][3][2])+"]"+"\n ["+str(dic[keys][4][0])+","\
                    +str(dic[keys][4][1])+","+str(dic[keys][4][2])+"]"
            #elif dic[keys]
            else : 
                values=="0000000"
        return values
    
    def launch(self):
        plt.ion()
        while 1:
            time.sleep(0.05)
            self.update()

# o=OptiMonitor_MPL()
# o.launch()
# o.update()
