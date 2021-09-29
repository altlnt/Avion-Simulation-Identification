#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 21:22:28 2020

@author: l3x
"""
import os
import pandas as pd
from pylab import *
from transforms3d import euler,quaternions
from scipy.interpolate import interp1d
from scipy.signal import resample
import pickle
import transforms3d as tf3d
import csv
import shutil, os

class Log():

    def __init__(self,ulg_path="",csv_path="",log_name=" ", drone_type="plane",is_sitl=True,roll_gain=1.0,pitch_gain=1.0):

        self.ulg_path=ulg_path
        self.csv_path=csv_path
        self.DATA=-1
        self.is_sitl=is_sitl
        self.drone_type=drone_type
        self.motor_gain_in_radsec=3500.0
        self.servo_gain=1.0
        self.log_name = log_name

        "gridsearch"
        self.roll_gain=roll_gain
        self.pitch_gain=pitch_gain
        return


    def log_to_csv(self):

        "file struct should be"

        "drone simulator"
        " ------ data"
        "         ------- csv"
        "                   empty"
        "         ------- ulg"
        "                  ------- .ulg file"



        path=self.ulg_path
        if not self.log_name.replace('.ulg','') in os.listdir(os.getcwd()):
            print(self.log_name)
            os.mkdir(path)
            shutil.move(os.getcwd()+'/'+self.log_name, path+'/'+self.log_name)

        print("Moving to...")
        os.chdir(path)
        if not 'donnees_brut' in os.listdir(path):
            os.mkdir("donnees_brut")
        print(path)
        print("Executing ulog2csv "+self.log_name+" -o donnees_brut")
        os.system('ulog2csv '+self.log_name+" -o donnees_brut")
        os.chdir('donnees_brut')
        print("Renaming files...")
        for file in os.listdir(os.getcwd()):
                if "actuator_outputs_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'actuators.csv'))
                if "battery_status_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'battery.csv'))
                if "vehicle_local_position_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'local_pos.csv'))

                if "gps_position_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'gps.csv'))
                if "actuator_controls_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'controls.csv'))
                if "angular_acceleration_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'angular_acc.csv'))
                if "angular_velocity_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'angular_vel.csv'))

                if "vehicle_attitude_groundtruth_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'attitude.csv'))
                elif "vehicle_attitude_0" in file:
                    os.system('cp '+file+' '+os.path.join(self.csv_path, 'attitude.csv'))
        os.chdir('../')

    def import_log(self,tcut=((-1.0,0.0),(1e15,1e100))):
        dpath=self.csv_path

        "check if there is already a DATA file in the directory"

        if dpath.endswith('/'):
            path=dpath
        else:
            path=dpath+'/'

        if 'DATA' in os.listdir(path):
            print("FOUND CSV FILE, LOADING")
            self.DATA=pd.read_pickle(path+"DATA")
            self.DATA=self.DATA.loc[(self.DATA.t.values>max(tcut[0])) & (self.DATA.t.values<min(tcut[1]))]
            return



        " we might use a tcut file if needed"
        if 'tcut.csv' in os.listdir(path):
            time_to_cut=pd.read_csv(os.path.join(path,"tcut.csv"))
            ltuples=[]
            for i in range(len(time_to_cut.time.values)-1):
                if i%2==0:
                    ltuples.append((time_to_cut.time.values[i],time_to_cut.time.values[i+1]))
            ltuples=tuple(ltuples)
            tcut_tup=ltuples
            print("FOUND TCUT FILE, OVERRIDING ARG TCUT")
        else:
            tcut_tup=tcut


        "READING CSV FILES"
        print(os.path.join(path))
        local_pos=pd.read_csv(os.path.join(path,"local_pos.csv"))
        act=pd.read_csv(os.path.join(path,"actuators.csv"))
        attitude=pd.read_csv(os.path.join(path,"attitude.csv"))
        ctrl=pd.read_csv(os.path.join(path,"controls.csv"))
        angax=pd.read_csv(os.path.join(path,"angular_acc.csv"))
        angvel=pd.read_csv(os.path.join(path,"angular_vel.csv"))

        "preparing the interpolation"
        "the initial time for the interpolation is the latest first data for"
        "each file"

        tmin,tmax=max([min(i['timestamp']) for i in [local_pos,attitude,act]]),min([max(i['timestamp'].values) for i in [local_pos,attitude,act]])



        attitude=attitude.loc[(attitude['timestamp']>tmin )& (attitude['timestamp']<tmax)]
        local_pos=local_pos.loc[(local_pos['timestamp']>tmin )& (local_pos['timestamp']<tmax)]
        act=act.loc[(act['timestamp']>tmin )& (act['timestamp']<tmax)]
        ctrl=ctrl.loc[(ctrl['timestamp']>tmin )& (ctrl['timestamp']<tmax)]
        angax=angax.loc[(angax['timestamp']>tmin )& (angax['timestamp']<tmax)]
        angvel=angvel.loc[(angvel['timestamp']>tmin )& (angvel['timestamp']<tmax)]


        highest_freq_DF=[local_pos,attitude,act,ctrl,angax,angvel][argmax([len(i) for i in [local_pos,attitude,act,ctrl,angax,angvel]])]
        ref_timestamp=highest_freq_DF['timestamp'].values


        DATA=pd.DataFrame(data=ref_timestamp,columns=['t'])

        "reinterpolating all signals"

        "transforming quat into euler and mat"
        "not the same for sitl and real logs"

        if self.is_sitl:
            attoffset=2
        else:
            attoffset=4

        att=asarray([euler.mat2euler(quaternions.quat2mat(attitude.values[i,attoffset:attoffset+4])) for i in range(len(attitude))]).reshape(-1,3)
        R=array([quaternions.quat2mat(attitude.values[i,attoffset:attoffset+4]) for i in range(len(attitude))]).reshape(-1,9)



        # for n_act in range(16):
        #     actinterp=interp1d(act.timestamp.values,act["output["+str(n_act)+"]"].values)
        #     DATA['act'+str(n_act)]=actinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        for n_ctrl in range(8):
            ctrlinterp=interp1d(ctrl.timestamp.values,ctrl["control["+str(n_ctrl)+"]"].values)
            DATA['ctrl'+str(n_ctrl)]=ctrlinterp(clip(DATA['t'].values,min(ctrl.timestamp.values),max(ctrl.timestamp.values)))

        axinterp=interp1d(local_pos.timestamp.values,local_pos.ax.values)
        DATA['ax']=axinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        ayinterp=interp1d(local_pos.timestamp.values,local_pos.ay.values)
        DATA['ay']=ayinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        azinterp=interp1d(local_pos.timestamp.values,local_pos.az.values)
        DATA['az']=azinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vxinterp=interp1d(local_pos.timestamp.values,local_pos.vx.values)
        DATA['vx']=vxinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vyinterp=interp1d(local_pos.timestamp.values,local_pos.vy.values)
        DATA['vy']=vyinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vzinterp=interp1d(local_pos.timestamp.values,local_pos.vz.values)
        DATA['vz']=vzinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))
    
        vxinterp=interp1d(local_pos.timestamp.values,local_pos.x.values)
        DATA['x']=vxinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vyinterp=interp1d(local_pos.timestamp.values,local_pos.y.values)
        DATA['y']=vyinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        vzinterp=interp1d(local_pos.timestamp.values,local_pos.z.values)
        DATA['z']=vzinterp(clip(DATA['t'].values,min(local_pos.timestamp.values),max(local_pos.timestamp.values)))

        for acc_ in range(3):
            angaccinterp=interp1d(angax.timestamp.values,angax["xyz["+str(acc_)+"]"].values)
            DATA['ang_acc'+str(acc_)]=angaccinterp(clip(DATA['t'].values,min(angax.timestamp.values),max(angax.timestamp.values)))


        for vel_ in range(3):
            angvelinterp=interp1d(angvel.timestamp.values,angvel["xyz["+str(vel_)+"]"].values)
            DATA['ang_vel'+str(vel_)]=angvelinterp(clip(DATA['t'].values,min(angvel.timestamp.values),max(angvel.timestamp.values)))

        for q_ in range(4):
            anginterp=interp1d(attitude.timestamp.values,attitude["q["+str(q_)+"]"].values)
            DATA['q['+str(q_)+"]"]=anginterp(clip(DATA['t'].values,min(attitude.timestamp.values),max(attitude.timestamp.values)))        

        for l_ in range(5):
            controlinterp=interp1d(act.timestamp.values,act["output["+str(l_)+"]"].values)
            DATA['output['+str(l_)+"]"]=controlinterp(clip(DATA['t'].values,min(act.timestamp.values),max(act.timestamp.values)))        


        attinterp=interp1d(attitude.timestamp.values,att.T)
        rsatt=attinterp(clip(DATA['t'].values,min(attitude.timestamp.values),max(attitude.timestamp.values)))


        Rinterp=interp1d(attitude.timestamp.values,R.T)

        rsR=Rinterp(clip(DATA['t'].values,min(attitude.timestamp.values),max(attitude.timestamp.values)))

        actinterp=interp1d(act.timestamp.values,act.values[:,2:10].T)
        ract=actinterp(clip(DATA['t'].values,min(act.timestamp.values),max(act.timestamp.values)))

        for i in range(8):
            DATA['act_'+str(i+1)]=ract[i]

        init_timestamp=DATA['t'].values[0]
        DATA['t']=(DATA['t']-DATA['t'].values[0])*1.0e-6


        DATA['roll'],DATA['pitch'],DATA['yaw']=rsatt
        DATA['init_timestamp ']=init_timestamp*ones(len(DATA))
        # kv_motor=960.0
        # pwmmin=1075.0
        # pwmmax=1950.0

        DATA['dt']=gradient(DATA.t.values)


        DATA['ax_alt']=gradient(DATA.vx,DATA.t)
        DATA['ay_alt']=gradient(DATA.vy,DATA.t)
        DATA['az_alt']=gradient(DATA.vz,DATA.t)

        DATA['R']=list(rsR.T)

        local_speed=[matmul(DATA['R'].values[i].reshape((3,3)).T,array([DATA['vx'].values[i],DATA['vy'].values[i],DATA['vz'].values[i]]).reshape(3,)) for i in range(len(DATA))]
        local_acc=[matmul(DATA['R'].values[i].reshape((3,3)).T,array([DATA['ax_alt'].values[i],DATA['ay_alt'].values[i],DATA['az_alt'].values[i]]).reshape(3,)) for i in range(len(DATA))]
        local_speed=vstack(local_speed)
        local_acc=vstack(local_acc)
        DATA["vx_local"],    DATA["vy_local"],    DATA["vz_local"]=local_speed.T
        DATA["ax_local"],    DATA["ay_local"],    DATA["az_local"]=local_acc.T


        print(DATA.t)
        "cutting the takeoff and landing moments"
        def keepit(value,tcut_tuples):
            for i in tcut_tuples:
                if min(i)<value and max(i)>value:
                    return False
            return True
        selection=[ keepit(i,tcut_tup) for i in DATA.t]
        DATA=DATA[selection]
        print("WRITING XLS FILE FOR FUTURE IMPORTS")
        DATA.to_pickle(self.ulg_path+"/DATA")
        self.DATA=DATA

        return


    def logs_to_ctrl_input(self,step_number):
        if self.drone_type=="plane":
            ctrl_input={'motors_speed_sp':array([self.motor_gain_in_radsec*self.DATA['ctrl3'].values[step_number]]),
                        'wing_angle_sp':array([self.roll_gain*self.servo_gain*self.DATA['ctrl0'].values[step_number], # aileron right
                                               -self.roll_gain*self.servo_gain*self.DATA['ctrl0'].values[step_number], # aileron left
                                               self.pitch_gain*self.servo_gain*self.DATA['ctrl1'].values[step_number], # elevon
                                               self.servo_gain*self.DATA['ctrl2'].values[step_number], # rudder
                                               self.servo_gain*self.DATA['ctrl4'].values[step_number], # airbrake
                                               self.servo_gain*self.DATA['ctrl4'].values[step_number]]), # airbrake
                        'motors_tilt_sp':-array([pi/2])}

        return ctrl_input
        
    def data_to_dic(self):
        self.DATA= pickle.load(open(self.ulg_path+"/DATA", "rb"))
        f = self.DATA

        dic_data={}

        dic_data['t']=f.t.values
        
        dic_data['acc[0]']=f.ax.values
        dic_data['acc[1]']=f.ay.values
        dic_data['acc[2]']=f.az.values

        dic_data['speed[0]']=f.vx.values
        dic_data['speed[1]']=f.vy.values
        dic_data['speed[2]']=f.vz.values
    
        dic_data['pos[0]']=f.x.values
        dic_data['pos[1]']=f.y.values
        dic_data['pos[2]']=f.z.values
        
        dic_data['omegadot[0]']=f.ang_acc0.values
        dic_data['omegadot[1]']=f.ang_acc1.values
        dic_data['omegadot[2]']=f.ang_acc2.values
        
        dic_data['omega[0]']=f.ang_vel0.values
        dic_data['omega[1]']=f.ang_vel1.values
        dic_data['omega[2]']=f.ang_vel2.values
        
        for q_ in range(4):
            dic_data["q["+str(q_)+"]"]=f["q["+str(q_)+"]"].values

        dic_data['forces[0]']=f.ax.values * 8.5
        dic_data['forces[1]']=f.ay.values * 8.5
        dic_data['forces[2]']=(f.az.values) * 8.5
        
        dic_data['torque[0]']=f.ang_acc0.values * 1.38
        dic_data['torque[1]']=f.ang_acc1.values * 0.84
        dic_data['torque[2]']=f.ang_acc2.values * 2.17
        
        # max_PWM=2000
        # min_PWM=1000
        mean_PWM_1=1527
        mean_PWM_2=1533
        dic_data['output[0]']=(f["output[0]"].values-mean_PWM_1)/500
        dic_data['output[1]']=(f["output[1]"].values-mean_PWM_1)/500
        dic_data['output[2]']=(f["output[2]"].values-mean_PWM_2)/500
        dic_data['output[3]']=(f["output[3]"].values-mean_PWM_2)/500
        dic_data['output[4]']=(f["output[4]"].values-1000)/1000

        dic_data['takeoff']=[1 if f.z.values[i]<-10 else 0 for i in range(len(f.t))]

        data = pd.DataFrame(dic_data)
        index_name=data[data['takeoff']==0].index

        data.drop(index_name,inplace=True)
        with open(os.path.join(self.ulg_path+"/log_real.csv"),'w',newline='') as log:
            data.to_csv(log)


logs=["log_5_2021-7-20-11-12-20", "log_6_2021-7-20-11-41-56"]
for i in range(len(logs)):
    os.chdir("/home/mehdi/Documents/identification_modele_avion/Logs/log_real/")
    log=logs[i]
    l=Log(ulg_path="/home/mehdi/Documents/identification_modele_avion/Logs/log_real/"+log,csv_path="/home/mehdi/Documents/identification_modele_avion/Logs/log_real/"+log+"/donnees_brut", log_name=log+'.ulg')
    l.log_to_csv()
    l.import_log()
    l.data_to_dic()
