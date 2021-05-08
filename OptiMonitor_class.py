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
        
        self.train_data=np.array([[0,10]])
        self.eval_data=np.array([[0,11]])

    def update(self):
        
        t=time.time()-self.t0
        y=10*np.exp(-0.01*t)
        
        self.train_data=np.row_stack([self.train_data,[t,y]])
        self.eval_data=np.row_stack([self.eval_data,[t,y+1.0]])
        
        self.train_score_curve.setData(self.train_data[:,0],self.train_data[:,1])
        self.eval_score_curve.setData(self.eval_data[:,0],self.eval_data[:,1])
        
        info="<div>t=%f , y=%f"%(t,y)
        info+="\n bonjour\n</div>"
        info+="<div>\n comment\n</div>"
        info+="<div>\n sava\n"
        info+="\n ajd\n</div>"

        self.label.setText(info)


    def launch(self):
        timer = pg.QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(50)
        if __name__ == '__main__':
            pg.mkQApp().exec_()
            print("YA")

o=OptiMonitor()
o.launch()