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
        self.speed=np.zeros(3)
        self.pos=np.zeros(3)
        
        #   Rotation
        self.omegadot=np.zeros(3)
        self.omega=np.zeros(3)
        self.q=np.array([1.0,0.0,0.0,0.0])

        # Dynamics params
        
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
    
    def compute_dynamics(self,joystick_input):
        # Faire le truc ici
        Forces,Torques=np.random.random_sample(3),np.random.random_sample(3)
        
        self.forces,self.torque=Forces,Torques
    
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
        list_to_write=[eval(i,scope) for i in keys]
        
        
        
        line_to_write=''
        for j,element in enumerate(list_to_write):
            if j!=0:
                line_to_write=line_to_write+","            
            line_to_write=line_to_write+str(element)
            
        line_to_write=line_to_write+"\n"  
        
        with open(os.path.join(self.data_save_path,"log.txt"),'a+') as f:
            print("Here: data")
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
while new_t-t0<5.0:
    new_t=time.time()
    Moteur.update_sim(new_t-t0,None)
    time.sleep(0.01)