#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:16:33 2021

@author: l3x
"""


import numpy as np
import time
import matplotlib.pyplot as plt

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


        # self.train_score_scat, = self.ax.scatter(self.t_data,self.train_data,label="train")
        # self.eval_score_scat, = self.ax.scatter(self.t_data,self.eval_data,label="eval")
        self.train_score_curve, = self.ax.plot(self.train_data,
                                               label="train",
                                               marker="x")
        self.eval_score_curve, = self.ax.plot(self.eval_data,
                                              label="eval",
                                              marker="o")
        
        self.ax.grid()
        self.ax.legend()

    def update(self):
        
        # self.t=time.time()-self.t0
        # self.y=10*np.exp(-0.01*self.t)
        
        # self.y_train,self.y_eval=self.y,self.y+1
        
        self.t_data=self.t_data+[self.t]
        self.train_data=self.train_data+[self.y_train]
        self.eval_data=self.eval_data+[self.y_eval]
        
        self.train_score_curve.set_data(self.t_data,self.train_data)
        self.eval_score_curve.set_data(self.t_data,self.eval_data)
        
        # self.train_score_scat.set_data(self.t_data,self.train_data)
        # self.eval_score_scat.set_data(self.t_data,self.eval_data)
        
        self.ax.relim()
        self.ax.autoscale_view()
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
