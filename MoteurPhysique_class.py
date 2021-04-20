#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:36:07 2021

@author: alex
"""

import numpy as np
import os
import time
from datetime import datetime
import transforms3d as tf3d
import dill as dill


dill.settings['recurse'] = True


savepath_base=os.path.join(os.getcwd(),"sim_logs",datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss"))

class MoteurPhysique():
    
    def __init__(self,save_path=savepath_base):
        
        try:
            os.mkdir(savepath_base)
        except:
            pass
        
        # Miscellaneous
        self.data_save_path=save_path
        self.last_t=0.0
        
        # Body state
        
        #   Translation
        self.forces,self.torque=np.zeros(3),np.zeros(3)
        self.acc=np.zeros(3)
        self.speed=np.array([1,0,0])
        self.pos=np.zeros(3)
        
        #   Rotation
        self.omegadot=np.zeros(3)
        self.omega=np.array([0,0,0])
        self.q=np.array([1.0,0.0,0.0,0.0])
        self.R=tf3d.quaternions.quat2mat(self.q)  

        # Dynamics params
        self.Dict_parametres = {"masse": 1.5 , \
                               "inertie": np.diag([0.19,0.15,0.15]),\
                               "alpha0" : np.array([3.44*np.pi/180,3.44*np.pi/180,0,0.0,0]),\
                               "alpha_stall" : 0.3391428111 ,                     \
                               "largeur_stall" : 30.0*np.pi/180,                  \
                               "wind" : np.array([0,0,0]),                        \
                               "g"    : np.array([0,0,-9.81]),                    \
                               "cp_list":[np.array([0,0.45,0], dtype=np.float),   \
                                      np.array([0,-0.45,0], dtype=np.float),      \
                                      np.array([-0.5,0.15,0], dtype=np.float),\
                                      np.array([-0.5,0.15,0], dtype=np.float),\
                                      np.array([0,0,0], dtype=np.float)]}
            
        self.Dict_variables = {"cd0sa" : 0.02,\
                               "cd0fp" : 0.02,\
                               "cl1fp" : 0.3, \
                               "cd1sa" : 0.5, \
                               "cl1sa" : 5.5, \
                               "cd1fp" : 0.3, \
                               "coeff_drag_shift":0.5, \
                               "coeff_lift_shift":0.5, \
                               "coef_lift_gain":0.5}
            
        self.Dict_etats     = {"position" : self.pos,    \
                               "vitesse" : self.speed,   \
                               "acceleration" : self.acc,\
                               "orientation" : self.q,   \
                               "vitesse_angulaire" : self.omega, \
                               "accel_angulaire" : self.omegadot}
            
        self.Dict_Var_Effort = {"Omega" :self.omega,\
                                "R": self.R.flatten(),\
                                "speed": self.speed, \
                                "Cd_list": np.array([0,0,0,0,0]), \
                                "Cl_list": np.array([0,0,0,0,0]), \
                                "Ct": 1e-4, \
                                "Cq": 1e-6, \
                                "Ch":1e-4,  \
                                "rotor_speed": 100}

        self.Dict_Commande = {"delta" : 0}
        # Test 
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
        r = np.array(( (1,0, 0), (0,c, -s),(0,s, c)) , dtype=np.float)
        return R * r
    
    def compute_alpha_sigma(dragDirection, liftDirection, frontward_Body, VelinLDPlane, alpha_0, alpha_s, delta_s): 
        calpha= np.vdot(dragDirection, frontward_Body)
        absalpha= -np.acos(calpha)
        signalpha = np.sign(np.vdot(liftDirection, frontward_Body)) 
        if np.linalg.norm(VelinLDPlane)>1e-7 :
            alpha = signalpha*absalpha 
        else :
            alpha=0
        if np.abs(alpha)>0.5*np.pi:
            if alpha>0 :alpha=alpha-np.pi 
            else: alpha=alpha+np.pi
                
        if alpha>=alpha_s+delta_s:
             sigma=0.0
        elif alpha>=alpha_s:
             sigma=0.5*(1.0+np.cos(np.pi*(alpha+alpha_0-alpha_s)/delta_s))
        elif alpha>=-alpha_s:
             sigma=1.0
        elif alpha>=-alpha_s-delta_s:
             sigma=0.5*(1.0+np.cos(np.pi*(alpha+alpha_0+alpha_s)/delta_s))
        else:
             sigma=0.0
        return float(alpha), float(sigma)

    
    def compute_dynamics(self,joystick_input):
        # Ouverture du fichier de fonction, on obtient une liste de fonction comme suit : 
            # 0 : VelinLDPlane_function
            # 1 : dragDirection_function
            # 2 : liftDirection_function
            # 3 : compute_alpha_sigma()
            # 4 : Coeff_function qui calcul les coeffs aéro pour toutes les surfaces tel que [Cl, Cd]
            # 5 : Effort_Aero qui renvoi un liste tel que [Force, Couple]
            
        Effort_function = dill.load(open('fichier_function','rb'))
        self.Dict_Commande["delta"] = 0  #joystick_input
        
        R_list         = [self.R, self.R, self.Rotation(self.R,45), self.Rotation(self.R,-45), self.R]
        v_W            = self.Dict_parametres["wind"]
        cp_list        = self.Dict_parametres['cp_list']
        alpha_0_list   = self.Dict_parametres["alpha0"]
        alpha_s        = self.Dict_parametres["alpha_stall"]
        delta_s        = self.Dict_parametres["largeur_stall"]
        frontward_Body = np.transpose(np.array([[1,0,0]]))
        
        cd1sa = self.Dict_variables["cd1sa"]            
        cl1sa = self.Dict_variables["cl1sa"]
        cd0sa = self.Dict_variables["cd0sa"]
        cd1fp = self.Dict_variables["cd1fp"]
        cd0fp = self.Dict_variables["cd0fp"]
        k0    = self.Dict_variables["coeff_drag_shift"]
        k1    = self.Dict_variables["coeff_lift_shift"]
        k2    = self.Dict_variables["coef_lift_gain"]
        cd = [0,0,0,0,0]
        cl = [0,0,0,0,0]
        p=0
        for cp in cp_list :          # Cette boucle calcul les coefs aéro pour chaque surface 
            VelinLDPlane   = Effort_function[0](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
            dragDirection  = Effort_function[1](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
            liftDirection  = Effort_function[2](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
            alpha, sigma = Effort_function[3](dragDirection, liftDirection, frontward_Body, VelinLDPlane, alpha_0_list[p], alpha_s, delta_s)
            cl[p], cd[p] = Effort_function[4](alpha, sigma, alpha_0_list[p],\
                                                                                                        self.Dict_Commande["delta"], \
                                                                                                        cl1sa, cd1fp,k1, k2, cd0fp, \
                                                                                                             cd0sa, cd1sa, cd1fp, k0)
            self.Dict_Var_Effort["Cl_list"] = cl
            self.Dict_Var_Effort["Cd_list"]= cd
            p+=1
                                                                                             

        Effort=Effort_function[5](self.omega, self.R.flatten(), self.speed.flatten(), v_W, self.Dict_Var_Effort["Cd_list"],\
                                  self.Dict_Var_Effort["Cl_list"], self.Dict_Var_Effort["Ct"], self.Dict_Var_Effort["Cq"], \
                                  self.Dict_Var_Effort["Ch"], self.Dict_Var_Effort["rotor_speed"])

        # Les calculs donnes des vecteurs lignes on transpose pour remettre en colone 
        self.forces,self.torque= self.R @ np.transpose(Effort[0].flatten()) +  self.Dict_parametres["g"], np.transpose(Effort[1]).flatten()   
     
    def update_state(self,dt):
        
        "update omega"
        
        J=np.eye(3)
        J_inv=np.eye(3)
        m=np.ones(3)
        
        new_omegadot=-np.cross(self.omega.T,np.matmul(J,self.omega.reshape((3,1))).flatten())
        new_omegadot=J_inv @ np.transpose(new_omegadot+self.torque)
        new_omega=self.omega+new_omegadot.flatten()*dt        

        self.omegadot=new_omegadot
        self.omega=new_omega
        
        "update q"
        
        qs,qv=self.q[0],self.q[1:]
        dqs=-0.5*np.dot(qv,self.omega)
        dqv=0.5*(qs*self.omega+np.cross(qv.T,self.omega.T).flatten())   
        dq=np.r_[dqs,dqv]
        new_q=self.q+dq*dt        
            
        R=tf3d.quaternions.quat2mat(new_q/tf3d.quaternions.qnorm(new_q))
        self.R=self.orthonormalize(R)
        self.q=tf3d.quaternions.mat2quat(R)      
        
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
              'torque[0]','torque[1]','torque[2]']
        
        t=self.last_t
        acc=self.acc
        speed=self.speed
        pos=self.pos
        omegadot=self.omegadot
        omega=self.omega
        q=self.q
        forces=self.forces
        torque=self.torque

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
        list_to_write=[(eval(i,scope) for i in keys)]
        
        
        
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
        self.compute_dynamics(joystick_input)
        self.update_state(dt)
        self.log_state()
        
        return

Moteur=MoteurPhysique()

t0=time.time()
new_t=t0
while new_t-t0<2.0:
    new_t=time.time()
    Moteur.update_sim(new_t-t0,None)
    time.sleep(0.01)