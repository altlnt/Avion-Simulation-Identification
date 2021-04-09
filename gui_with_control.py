#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:04:00 2021

@author: alex
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import socket


class viewer:
    def __init__(self):
        
        self.app = pg.mkQApp("Sim GUI")
        ####################################################################
        
        #           Initialisation du widget de visualisation de la rotation
        "3D rotation"       
        self.w_rot = gl.GLViewWidget()
        self.w_rot.opts['distance'] = 20
        self.w_rot.show()
        self.w_rot.setWindowTitle(' ROTATION')
        
        self.g_rot = gl.GLGridItem()
        self.w_rot.addItem(self.g_rot)
        
        #                   Création de la géométrie (on pourrait mettre un mesh de drone pour être plus "réaliste")
        
        
        self.verts = np.array([
            [1, 0, 0],
            [-1, 0.5, 0.5],
            [-1, -0.5, -0.5],
            [-1, -0.5, 0.5],
            [-1, 0.5, -0.5]
        ])
        self.faces = np.array([
            [0, 1, 2],
            [0, 3, 4]
        
        ])
        self.colors = np.array([
            [1, 0, 0, 0.3],
            [0, 1, 0, 0.3]])
        
        self.m1 = gl.GLMeshItem(vertexes=self.verts, faces=self.faces, faceColors=self.colors, smooth=False)
        self.m1.translate(0, 0, 0)
        self.m1.setGLOptions('additive')
        self.w_rot.addItem(self.m1)
        
        #                   Callback update de l'affichage de la rotation 
        
        self.phase=0
        ####################################################################
        
        #           Initialisation du widget de visualisation de la translation
        
        "2d translation"
        
        self.w_translation = gl.GLViewWidget()
        self.w_translation.opts['distance'] = 20
        self.w_translation.show()
        self.w_translation.setWindowTitle('Translation')
        
        self.g_translation = gl.GLGridItem()
        self.w_translation.addItem(self.g_translation)
        
        
        
        self.pos = [[0,0,0]]
        
        self.Msize=0.6
        self.Mcolor=(1.0,1.0,0.0,1.0)
        self.size=0.3
        self.color=(1.0,0.0,0.0,0.5)
        
        self.cols=[self.Mcolor]
        self.sizes=[self.Msize]
        
        
        self.sp1 = gl.GLScatterPlotItem(pos=np.array(self.pos), size=np.array(self.sizes),color=np.array(self.cols), pxMode=False)
        self.w_translation.addItem(self.sp1)
        #                   Callback update de l'affichage de la translation 
        return
    
    def update_rot(self):
        
        ## update rotation 
        self.m1.resetTransform()
    
        ################### récupérer la rotation et la feeder ici 
     
        self.m1.rotate(1.0,0.0,np.cos(self.phase),-np.sin(self.phase),"quaternion")
        ###################
        return
            
    def update_translation(self):
        
        self.cols[-1]=self.color
        self.cols.append(self.Mcolor)
        self.sizes[-1]=self.size
        self.sizes.append(self.Msize)
    
    
    
        ################### récupérer la position du modèle et la feeder ici
    
        self.new_pos=[np.cos(self.phase),-np.sin(self.phase),np.sin(0.5*self.phase)]
    
        ###################
    
    
    
        self.pos.append(self.new_pos)
        self.sp1.setData(pos=np.array(self.pos), color=np.array(self.cols),size=np.array(self.sizes))


        # Callback l'appli 
        return
    def update(self):
        self.update_rot()
        self.update_translation()
        self.phase -= 0.1
        return
    
    def launch(self):
        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.update)
        self.t.start(50)
        pg.mkQApp().exec_()
        return
    
import multiprocessing as mp
G=viewer()
G.launch()
