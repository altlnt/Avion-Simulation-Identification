#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:06:54 2021

@author: l3x
"""
from threading import Thread
# import threading
# import thread
def fonction_1():
    while True:
        print("1")
    return


def fonction_2():
    while True:
        print("2")
    return

TA=Thread(None,fonction_1)
TB=Thread(None,fonction_2)

TA.start()
TB.start()