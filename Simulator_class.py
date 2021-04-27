#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:33:53 2021

@author: alex
"""
import numpy as np
from Gui_class import Viewer
from MoteurPhysique_class import MoteurPhysique
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import time 

class Simulator():
    
    def __init__(self):
        
        self.viewer=Viewer()
        self.moteur_phy=MoteurPhysique()
        self.t0=-1
        
    def update(self):
        
        joystick_input=np.array([self.viewer.joystick_L_horizontal,
        self.viewer.joystick_L_vertical,
        self.viewer.joystick_R_horizontal,
        self.viewer.joystick_R_vertical])        
        
        
        self.moteur_phy.update_sim(time.time()-self.t0, joystick_input)
        
        self.viewer.target_q=self.moteur_phy.q
        self.viewer.target_pos=self.moteur_phy.pos
        
        # phase=time.time()-self.t0
        # self.viewer.target_q=np.array([1.0,0.0,np.cos(phase),-np.sin(phase)])
        # self.viewer.target_pos=[np.cos(phase),-np.sin(phase),np.sin(0.5*phase)]

        self.viewer.update_rot()
        self.viewer.update_translation()
        self.viewer.update_joysticks()        
        
    def launch_sim(self):
        
        #self.viewer.w_rot.show()
        self.viewer.w_translation.show()
        self.viewer.mw.show()

        self.t0=time.time()
        self.viewer.t = QtCore.QTimer()
        self.viewer.t.timeout.connect(self.update)
        self.viewer.t.start(50)
        pg.mkQApp().exec_()        
        
        return
    
    
S=Simulator()
S.launch_sim()