#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:58:19 2021

@author: alex
"""

# Import des libs

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np


# Initialisation de l'appli 

app = pg.mkQApp("Sim GUI")

####################################################################

#           Initialisation du widget de visualisation de la rotation


"3D rotation"

w_rot = gl.GLViewWidget()
w_rot.opts['distance'] = 20
w_rot.show()
w_rot.setWindowTitle(' ROTATION')

g_rot = gl.GLGridItem()
w_rot.addItem(g_rot)

#                   Création de la géométrie (on pourrait mettre un mesh de drone pour être plus "réaliste")


verts = np.array([
    [1, 0, 0],
    [-1, 0.5, 0.5],
    [-1, -0.5, -0.5],
    [-1, -0.5, 0.5],
    [-1, 0.5, -0.5]
])
faces = np.array([
    [0, 1, 2],
    [0, 3, 4]

])
colors = np.array([
    [1, 0, 0, 0.3],
    [0, 1, 0, 0.3]

])

m1 = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=False)
m1.translate(0, 0, 0)
m1.setGLOptions('additive')
w_rot.addItem(m1)

#                   Callback update de l'affichage de la rotation 

def update_rot():

    ## update rotation 
    global phase
    m1.resetTransform()

    ################### récupérer la rotation et la feeder ici 
 
    m1.rotate(1.0,0.0,np.cos(phase),-np.sin(phase),"quaternion")
    ###################


####################################################################

#           Initialisation du widget de visualisation de la translation

"2d translation"

w_translation = gl.GLViewWidget()
w_translation.opts['distance'] = 20
w_translation.show()
w_translation.setWindowTitle('Translation')

g_translation = gl.GLGridItem()
w_translation.addItem(g_translation)



pos = [[0,0,0]]

Msize=0.6
Mcolor=(1.0,1.0,0.0,1.0)
size=0.3
color=(1.0,0.0,0.0,0.5)

cols=[Mcolor]
sizes=[Msize]


sp1 = gl.GLScatterPlotItem(pos=np.array(pos), size=np.array(sizes),color=np.array(cols))
w_translation.addItem(sp1)

#                   Callback update de l'affichage de la translation 


def update_translation():
    global phase,pos,sizes,cols

    cols[-1]=color
    cols.append(Mcolor)
    sizes[-1]=size
    sizes.append(Msize)



    ################### récupérer la position du modèle et la feeder ici

    new_pos=[np.cos(phase),-np.sin(phase),np.sin(0.5*phase)]

    ###################



    pos.append(new_pos)
    sp1.setData(pos=np.array(pos), color=np.array(cols),size=np.array(sizes))


# Callback l'appli 


def update():
    global phase
    update_rot()
    update_translation()
    phase -= 0.1


t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)
phase=0


if __name__ == '__main__':
    pg.mkQApp().exec_()
