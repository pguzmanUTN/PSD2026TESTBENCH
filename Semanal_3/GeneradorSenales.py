#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal as sp
"""
Created on Wed Mar 25 10:25:06 2026

@author: pedro
"""
#%%
N    = 1000 #Muestras
fs   = N #Resolucion espectral normalizada

ff   = 10 # Hz
vmax = 5 #Volt
dc   = 1 #Volt
px   = np.pi/2 #rads

#%%
def  mi_funcion ( vmax = 1, dc = 0, ff = 1, ph = 0, nn = 100, fs = 50, type = 'sine', duty = 0.5) :
    
    tt = np.arange(start = 0, step = 1/fs, stop = N/fs)
    if type == 'sine' :
        xx = vmax*np.sin(2*np.pi*ff*tt+ph)+dc
    elif type == 'sawtooth' :
        xx = vmax*sp.sawtooth(2*np.pi*ff*tt+ph)+dc
    elif type == 'triangle' :
        xx = vmax*sp.sawtooth(2*np.pi*ff*tt+ph,width = 0.5)+dc
    elif type == 'invsawtooth' :
        xx = vmax*sp.sawtooth(2*np.pi*ff*tt+ph, width = 0)+dc
    elif type == 'square' :
        xx = vmax*sp.square(2*np.pi*ff*tt+ph, duty = duty)+dc
    
    return tt, xx

#%%
tt, xx = mi_funcion( vmax = vmax, dc = dc, ff = ff, ph= px, nn = N, fs = fs, type = 'square', duty = 0.75)


plt.xlim([0,5/ff]) #Para que me muestre dos ciclos
plt.plot(tt, xx, marker = 'X', color = 'r', linestyle = '-')
plt.grid()
plt.title("Salida Generador de Senales")
plt.ylabel("Amplitud [V]")
plt.xlabel("Tiempo [s]")