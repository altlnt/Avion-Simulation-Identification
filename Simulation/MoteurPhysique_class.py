#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:36:07 2021

@author: alex
"""

import sys
sys.path.append('../')
import numpy as np
import os
import time
from datetime import datetime
import transforms3d as tf3d
import dill as dill
import json
from collections import OrderedDict

dill.settings['recurse'] = True


class MoteurPhysique():
    def __init__(self,called_from_opti=False,T_init=1.0):
        
        
        self.save_path_base=os.path.join("../Logs",datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss"))

        if not called_from_opti:
            os.makedirs(self.save_path_base)
 
        
        # Miscellaneous
        self.data_save_path=self.save_path_base
        self.last_t=0.0
        self.T_init=T_init
        # Body state
        
        #   Translation
        self.forces,self.torque=np.zeros(3),np.zeros(3)
        self.grad_forces, self.grad_torque = np.zeros((3,10)), np.zeros((3,10))
        self.acc=np.zeros(3)
        self.speed=np.array([0.01,0,0])
        self.pos=np.zeros(3)
        
        #   Rotation
        self.omegadot=np.zeros(3)
        self.omega=np.array([0,0,0])
        self.q=np.array([1.0,0.0,0.0,0.0])
        self.R=tf3d.quaternions.quat2mat(self.q) 
        self.moy_rotor_speed = 200
        self.takeoff =0
        
        self.Effort_function = dill.load(open('../Simulation/function_moteur_physique','rb'))
        self.joystick_input = [0,0,0,0,0]
        self.joystick_input_log= [0,0,0,0,0]
        
        ####### Dictionnaire des paramètres du monde 
        self.Dict_world     = {"wind" : np.array([0,0,0]),                        \
                               "g"    : np.array([0,0,9.81]),                    \
                               }
        ####### Dictionnaire des paramètres pour l'optimisation
        self.Dict_variables = {"masse": 2.5 , \
                               "inertie": np.diag([0.2,0.15,0.15]),\
                               "aire" : [0.03* 1.292 * 0.5, 0.03* 1.292 * 0.5, 0.01* 1.292 * 0.5, 0.01* 1.292 * 0.5, 0.04* 1.292 * 0.5],\
                               "cp_list": [np.array([0,0.45,0],       dtype=np.float).flatten(), \
                                           np.array([0,-0.45,0],      dtype=np.float).flatten(), \
                                           np.array([-0.5,0.15,0],    dtype=np.float).flatten(),\
                                           np.array([-0.5,-0.15,0],   dtype=np.float).flatten(),\
                                           np.array([0,0,0],          dtype=np.float).flatten()],
                               "alpha0" : np.array([0.06,0.06,0,0,0.06]),\
                               "alpha_stall" : 0.3391428111 ,                     \
                               "largeur_stall" : 30.0*np.pi/180,                  \
                               "cd0sa" : 0.045,\
                               "cd0fp" : 0.045,\
                               "cd1sa" : 4.55, \
                               "cl1sa" : 5, \
                               "cd1fp" : 2.5, \
                               "coeff_drag_shift": 0.5, \
                               "coeff_lift_shift": 0.5, \
                               "coeff_lift_gain": 0.5,\
                               "Ct": 2.5e-5, \
                               "Cq": 1e-8, \
                               "Ch": 1e-4}
            
        # self.Dict_variables = OrderedDict(sorted(self.Dict_variables.items(), key=lambda t: t[0]))
            
        # Dictionnaire des états pour la jacobienne
        self.Theta = ['alpha_stall',
                        'largeur_stall',
                        'cd1sa',
                        'cl1sa',
                        'cd0sa',
                        'cd1fp',
                        'cd0fp',
                        'coeff_drag_shift',
                        'coeff_lift_shift',
                        'coeff_lift_gain']
        
            
        # Dictionnaires des états
        self.Dict_etats     = {"position" : self.pos,    \
                               "vitesse" : self.speed,   \
                               "acceleration" : self.acc,\
                               "orientation" : self.q,   \
                               "vitesse_angulaire" : self.omega, \
                               "accel_angulaire" : self.omegadot,\
                               "alpha" : 0.0}
            
        self.Dict_Var_Effort = {"Omega" :self.omega,\
                                "speed": self.speed, \
                                "Cd_list": np.array([0,0,0,0,0]), \
                                "Cl_list": np.array([0,0,0,0,0]), \
                                }
        
        self.Dict_Commande = {"delta" : 0,\
                              "rotor_speed" : self.moy_rotor_speed }
 
    
    
        self.SaveDict={} 
        self.called_from_opti=called_from_opti
        if not called_from_opti:
            for dic in [self.Dict_world,self.Dict_variables]:
                keys=dic.keys() 
                for key in keys:
                    self.SaveDict[key]=np.array(dic[key]).tolist()
            self.SaveDict['mean_rotor_speed']=self.moy_rotor_speed
            with open(os.path.join(self.save_path_base,'params.json'), 'w') as fp:
                json.dump(self.SaveDict, fp)
        
            print(self.data_save_path)
    
    def orthonormalize(self,R_i):
        R=R_i
        R[:,0]/=np.linalg.norm(R[:,0])
        R[:,1]=R[:,1]-np.dot(R[:,0].flatten(),R[:,1].reshape((3,1)))*R[:,0]
        R[:,1]/=np.linalg.norm(R[:,1])
        R[:,2]=np.cross(R[:,0],R[:,1])
        R[:,2]/=np.linalg.norm(R[:,2])
        return R

    def Rotation(self,R,angle):
        c, s = np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)
        r = np.array(( (1,0, 0), (0,c, s),(0,-s, c)) , dtype=np.float)
        return R @ r
    
    def EulerAngle(self, q):
        # Calcul les angles roll, pitch, yaw en fonction du quaternion, utiliser uniquement pour le plot
        sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
        cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
        if (abs(sinp) >= 1):
            pitch = np.sign(np.pi/ 2, sinp) 
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2]);
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll,pitch,yaw])
    
    def compute_dynamics(self,joystick_input,t, compute_gradF=None):
        # Ouverture du fichier de fonction, on obtient une liste de fonction comme suit : 
            # 0 : VelinLDPlane_function
            # 1 : dragDirection_function
            # 2 : liftDirection_function
            # 3 : compute_alpha_sigma()
            # 4 : Coeff_function qui calcul les coeffs aéro pour toutes les surfaces tel que [Cl, Cd]
            # 5 : Effort_Aero qui renvoi un liste tel que [Force, Couple]
            # 6 : Gradient de sigma (pas utilisé dans cette version)
            # 7-8 : Gradient de Cl, et de Cd respectivement, renvoi un vecteur 
            # 9 : Sommes des gradients des efforts pour chaques ailes, et moteurs
            
        T_init=self.T_init    # Temps pendant laquelle les forces ne s'appliquent pas sur le drone
        self.joystick_input_log= joystick_input

        for q,i in enumerate(joystick_input):      # Ajout d'une zone morte dans les commandes 
            if abs(i)<40 :
                self.joystick_input[q] = 0
            elif q==0 or q==2 :
                self.joystick_input[q] = 0
            elif q==3 : 
                self.joystick_input[q] = joystick_input[q]/250 *  self.moy_rotor_speed
            else:
                self.joystick_input[q] = joystick_input[q] * 15 *np.pi/180 / 250 
        # Mise à niveau des commandes pour etre entre -15 et 15 degrés 
         # (l'input est entre -250 et 250 initialement)
        self.Dict_Commande["delta"] = np.array([self.joystick_input[0], -self.joystick_input[0], \
                                                (self.joystick_input[1] - self.joystick_input[2]) \
                                                , (self.joystick_input[1] + self.joystick_input[2]) , 0]) \
            
        self.Dict_Commande["rotor_speed"] =  self.moy_rotor_speed + self.joystick_input[3]
                                                                     
        R_list         = [self.R, self.R, self.Rotation(self.R, 45), self.Rotation(self.R,-45), self.R]
        v_W            = self.Dict_world["wind"]
        frontward_Body = np.transpose(np.array([[1,0,0]]))
        A_list         = self.Dict_variables["aire"]
        alpha_0_list   = self.Dict_variables["alpha0"]
        alpha_s        = self.Dict_variables["alpha_stall"]
        delta_s        = self.Dict_variables["largeur_stall"]  
        cp_list        = self.Dict_variables['cp_list']
        cd1sa = self.Dict_variables["cd1sa"]            
        cl1sa = self.Dict_variables["cl1sa"]
        cd0sa = self.Dict_variables["cd0sa"]
        cd1fp = self.Dict_variables["cd1fp"]
        cd0fp = self.Dict_variables["cd0fp"]
        k0    = self.Dict_variables["coeff_drag_shift"]
        k1    = self.Dict_variables["coeff_lift_shift"]
        k2    = self.Dict_variables["coeff_lift_gain"]

        if compute_gradF==None:
            if (t)<T_init:
                self.forces= np.array([0,0,0]) 
                self.torque= np.array([0,0,0])
                if not self.called_from_opti:
                    print("Début des commandes dans :", T_init-t)
            else: 
                alpha_list=[0,0,0,0,0]
                for p, cp in enumerate(cp_list) :          # Cette boucle calcul les coefs aéro pour chaque surface 
                    VelinLDPlane   = self.Effort_function[0](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                    dragDirection  = self.Effort_function[1](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                    liftDirection  = self.Effort_function[2](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                    alpha_list[p] = self.Effort_function[3](dragDirection, liftDirection, frontward_Body, VelinLDPlane)

                self.Dict_etats['alpha'] = alpha_list[-1]

                forces, torque = self.Effort_function[4](A_list, self.omega, self.R.flatten(), self.speed.flatten(),\
                                                  v_W, cp_list,alpha_list, alpha_0_list,\
                                                   alpha_s, self.Dict_Commande["delta"], \
                                                   delta_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, \
                                                   cd0sa, cd1sa, \
                                                  self.Dict_variables["Ct"], self.Dict_variables["Cq"], \
                                                  self.Dict_variables["Ch"],self.Dict_Commande["rotor_speed"])

                self.forces= self.R @ np.transpose(forces.flatten()) +  self.Dict_variables["masse"] *self.Dict_world["g"]
                self.torque = np.transpose(torque).flatten()  
                if self.takeoff==0:
                    self.forces[2]=min(self.forces[2],0)
                    
################## Calcul du gradient #################
        else:
            if (t)<T_init:
                 self.grad_forces, self.grad_torque=np.zeros((3,len(self.Theta))),np.zeros((3,len(self.Theta)))
            else: 
                alpha_list=[0,0,0,0,0]
                for p, cp in enumerate(cp_list) :          
                    VelinLDPlane   = self.Effort_function[0](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                    dragDirection  = self.Effort_function[1](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                    liftDirection  = self.Effort_function[2](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                    alpha_list[p] = self.Effort_function[3](dragDirection, liftDirection, frontward_Body, VelinLDPlane)
                    
########## Calcul compact à partir d'une liste des angles d'attaques, le calcul des coeffs aéro est compris dedans
                self.grad_forces = self.Effort_function[5](A_list, self.omega, self.R.flatten(), self.speed.flatten(),\
                                                  v_W, cp_list,alpha_list, alpha_0_list,\
                                                   alpha_s, self.Dict_Commande["delta"], \
                                                   delta_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, \
                                                   cd0sa, cd1sa, \
                                                  self.Dict_variables["Ct"], self.Dict_variables["Cq"], \
                                                  self.Dict_variables["Ch"],self.Dict_Commande["rotor_speed"])[0]
                self.grad_torque = self.Effort_function[5](A_list, self.omega, self.R.flatten(), self.speed.flatten(),\
                                                  v_W, cp_list,alpha_list, alpha_0_list,\
                                                   alpha_s, self.Dict_Commande["delta"], \
                                                   delta_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, \
                                                   cd0sa, cd1sa, \
                                                  self.Dict_variables["Ct"], self.Dict_variables["Cq"], \
                                                  self.Dict_variables["Ch"],self.Dict_Commande["rotor_speed"])[1]
                for col in range((np.size(self.Theta))):
                    self.grad_forces[:,col]= self.R @ np.array((self.grad_forces[0,col],self.grad_forces[1,col],self.grad_forces[2,col]))
                    self.grad_torque[:,col]= self.R @ np.array((self.grad_torque[0,col],self.grad_torque[1,col],self.grad_torque[2,col]))
                    if self.takeoff==0:
                        self.grad_forces[:,col]=self.R @ np.array((self.grad_forces[0,col],self.grad_forces[1,col],0))
                        
    def update_state(self,dt):
        
        "update omega"
        
        J=self.Dict_variables['inertie']
        J_inv=np.linalg.inv(J)
        m=np.ones(3) * self.Dict_variables['masse']
        
        new_omegadot=-np.cross(self.omega.T,np.matmul(J,self.omega.reshape((3,1))).flatten())
        new_omegadot=J_inv @ np.transpose(new_omegadot+self.torque)
        new_omega=self.omega+new_omegadot.flatten()*dt        

        self.omegadot=new_omegadot
        self.omega=new_omega
        
        if abs(self.pos[2])<0.001 and self.takeoff==0:
            self.omega=np.array([0,max(self.omega[1],0),self.omega[2]]) 
        elif abs(self.pos[2])>0.001 and self.takeoff==0:
            print("Décollage effectué")
            self.takeoff = 1 
        
        "update q"
        
        qs,qv=self.q[0],self.q[1:]
        dqs=-0.5*np.dot(qv,self.omega)
        dqv=0.5*(qs*self.omega+np.cross(qv.T,self.omega.T).flatten())   
        dq=np.r_[dqs,dqv]
        new_q=self.q+dq*dt        
            
        R=tf3d.quaternions.quat2mat(new_q/tf3d.quaternions.qnorm(new_q))
        self.R=self.orthonormalize(R)
        self.q=tf3d.quaternions.mat2quat(R)    
        #print(tf3d.quaternions.qnorm(self.q))
        "update forces"
                
        self.acc=self.forces/m
        self.speed=self.speed+self.acc*dt
        self.pos=self.pos+self.speed*dt        



    def log_state(self):
        
        keys=['t','acc[0]','acc[1]','acc[2]',
              'speed[0]','speed[1]','speed[2]',
              'pos[0]','pos[1]','pos[2]',
              'omegadot[0]','omegadot[1]','omegadot[2]',
              'omega[0]','omega[1]','omega[2]',
              'q[0]','q[1]','q[2]','q[3]',
              'forces[0]','forces[1]','forces[2]',
              'torque[0]','torque[1]','torque[2]','alpha',
              'joystick[0]','joystick[1]','joystick[2]',  
              'joystick[3]','takeoff']
        
        t=self.last_t
        acc=self.acc
        speed=self.speed
        pos=self.pos
        omegadot=self.omegadot
        omega=self.omega
        q=self.q
        forces=self.forces
        torque=self.torque
        alpha=self.Dict_etats['alpha']
        euler = self.EulerAngle(q) * 180/np.pi
        joystick_input = self.joystick_input_log
        TK=self.takeoff

        if 'log.txt' not in os.listdir(self.data_save_path):
            print("Here: Init")
            first_line=""
            for j,key in enumerate(keys):
                if j!=0:
                    first_line=first_line+","
                first_line=first_line+key
                
            first_line=first_line+"\n"
            with open(os.path.join(self.data_save_path,"log.txt"),'a') as f:
                f.write(first_line)
        
        scope=locals()
        list_to_write=[t,acc[0],acc[1],acc[2],
              speed[0],speed[1],speed[2],
              pos[0],pos[1],pos[2],
              omegadot[0],omegadot[1],omegadot[2],
              omega[0],omega[1],omega[2],
              q[0],q[1],q[2],q[3],
              forces[0],forces[1],forces[2],
              torque[0],torque[1],torque[2], alpha, 
              joystick_input[0], joystick_input[1],
              joystick_input[2], joystick_input[3], TK]
        
        
        
        line_to_write=''
        for j,element in enumerate(list_to_write):
            if j!=0:
                line_to_write=line_to_write+","            
            line_to_write=line_to_write+str(element)
            
        line_to_write=line_to_write+"\n"  
        
        with open(os.path.join(self.data_save_path,"log.txt"),'a+') as f:
            #print("Here: data")
            f.write(line_to_write)        
            
    def update_sim(self,t,joystick_input):
        
        dt=t-self.last_t
        self.last_t=t
        self.compute_dynamics(joystick_input,self.last_t)
        self.update_state(dt)
        self.log_state()
        
        return

# Moteur=MoteurPhysique()

# t0=time.time()
# new_t=t0
# while new_t-t0<2.0:
#     new_t=time.time()
#     Moteur.update_sim(new_t-t0,None)
#     time.sleep(0.01)