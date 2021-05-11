#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:16:33 2021

@author: l3x
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import time 


class OptiMonitor():
    def __init__(self,name=None):
        
        self.name=name if name!=None else str(int(time.time()))
        
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle(self.name)
        self.plot_item= self.win.addPlot()
        self.plot_item.setDownsampling(mode='peak')
        self.plot_item.setLabel('bottom', 'epoch')

        self.label= pg.LabelItem(justify='right')
        self.win.addItem(self.label)

        self.train_score_curve = self.plot_item.plot()
        self.eval_score_curve = self.plot_item.plot()
        
        
        
        self.t0=time.time()
        self.t,self.y_train,self.y_eval=0,10,11
        self.train_data=np.array([[self.t,self.y_train]])
        self.eval_data=np.array([[self.t,self.y_eval]])

    def update(self):
        
        # self.t=time.time()-self.t0
        # self.y=10*np.exp(-0.01*self.t)
        
        self.train_data=np.row_stack([self.train_data,[self.t,self.y_train]])
        self.eval_data=np.row_stack([self.eval_data,[self.t,self.y_eval]])
        
        self.train_score_curve.setData(self.train_data[:,0],self.train_data[:,1])
        self.eval_score_curve.setData(self.eval_data[:,0],self.eval_data[:,1])
        
        info="<div>t=%f , y=%f"%(self.t,self.y_train)
        info+="\n bonjour\n</div>"
        info+="<div>\n comment\n</div>"
        info+="<div>\n sava\n"
        info+="\n ajd\n</div>"

        self.label.setText(info)
        self.win.update()
        return

    def launch(self):
        timer = pg.QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(50)
        pg.mkQApp().exec_()

# o=OptiMonitor()
# # o.launch()
# o.update()
