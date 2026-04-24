#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal as sp
import time
from scipy.fft import fft, ifft

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

N=16384
fs   = 1e3 
n = np.arange(start = 0, step = 1/fs, stop = N/fs)
ff = 50
t, x = mi_funcion( vmax = 1, dc = 0, ff = ff, ph = np.pi/4, nn = N, fs = fs, type = 'square', duty = 0.25)
plt.plot(t,x)
plt.grid()

def DFT(x):
    
    N=len(x)
    n = np.arange(N)
    k=n.reshape((N,1)) #Vector fila a Vector columna
    e=np.exp(-2j*np.pi*k*n/N)
    Xk=np.dot(x,e) #Producto Punto.
    
    return Xk 


inicio = time.time()

Xk = DFT(x)

fin = time.time()

timedft = fin - inicio

print(f"Tiempo de ejecución DFT: {fin - inicio:.9f} segundos")


inicio = time.time()

Xft = fft(x)

fin = time.time()

timefft = fin - inicio

print(f"Tiempo de ejecución FFT: {fin - inicio:.9f} segundos")

#print(f"Ganancia de Procecsamiento: {timedft/timefft:.9f}")
f  = np.linspace(0, fs, N)

plt.figure()
plt.title("Modulo")
plt.plot(f , np.abs(Xk)/len(Xk), label= "DFT" , marker = 'X')
plt.plot(f, np.abs(Xft)/len(Xft), label= "FFT", marker = 'o')
plt.xlim(0, fs/2)
plt.grid()
plt.legend()

th = 0.005
thv = ((np.abs(Xk)/len(Xk)) > th).astype(int)
plt.figure()
plt.title("Fase")
plt.plot(f,np.angle(Xk,deg=True)*thv, label= "DFT", marker = 'X' )
plt.plot(f,np.angle(Xft,deg=True)*thv, label= "FFT", marker = 'o')
plt.xlim(0, fs/2)
plt.grid()
plt.legend()
phase = np.angle(Xk,deg=True)


plt.figure()
plt.title("Modulo")
plt.plot(f,np.abs(Xk)/len(Xk), label= "DFT", marker = 'X')
plt.plot(f,np.abs(Xft)/len(Xft), label= "FFT", marker = 'o')
plt.xlim(0, fs/2)
plt.grid()
plt.legend()

plt.figure()
plt.title("Real")
plt.plot(f,np.real(Xk)/len(Xk), label= "DFT" , marker = 'X')
plt.plot(f,np.real(Xft)/len(Xft), label= "FFT", marker = 'o')
plt.xlim(0, fs/2)
plt.grid()
plt.legend()

plt.figure()
plt.title("Imaginario")
plt.plot(f,np.imag(Xk)/len(Xk), label= "DFT" , marker = 'o')
plt.plot(f,np.imag(Xft)/len(Xft), label= "FFT", marker = 'o')
plt.xlim(0, fs/2)
plt.grid()
plt.legend()

