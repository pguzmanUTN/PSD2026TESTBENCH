#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal as sp
import time
from scipy.fft import fft, ifft
import scipy.signal as signal

#Funcion del generador de se;ales
def  mi_funcion ( vmax = 1, dc = 0, ff = 1, ph = 0, nn = 100, fs = 50, type_s = 'sine', duty = 0.5, vmed=5e-3, vsigma=0.5e-3) :
    
    tt = np.arange(start = 0, step = 1/fs, stop = nn/fs)
    if   type_s == 'sine' :
        xx = vmax*np.sin(2*np.pi*ff*tt+ph)+dc
    elif type_s == 'sawtooth' :
        xx = vmax*sp.sawtooth(2*np.pi*ff*tt+ph)+dc
    elif type_s == 'triangle' :
        xx = vmax*sp.sawtooth(2*np.pi*ff*tt+ph,width = 0.5)+dc
    elif type_s == 'invsawtooth' :
        xx = vmax*sp.sawtooth(2*np.pi*ff*tt+ph, width = 0)+dc
    elif type_s == 'square' :
        xx = vmax*sp.square(2*np.pi*ff*tt+ph, duty = duty)+dc
    elif type_s == 'noise' :
        xx = np.random.normal(vmed, vsigma, len(tt))+dc
    else :
        print("Tipo de funcion no existente o incorrecta")
    
    return tt, xx

def Quant(VFS, x, B) :
    
    q = VFS/(2**B)
    SQ = np.round(x/q)*q
    SQ = np.clip(SQ,a_min=-q*(2**(B-1)) , a_max=q*(2**(B-1)-1))
    print(f"Cuanto={q} para Vref={VFS} y cantidad de bits = {B}")
    return SQ

def  mi_generador_ruidoso ( Psine = 1,R = 1 ,SNRdB=10, ff = 1, ph = 0, nn = 100, fs = 50) :
    vmax=np.sqrt(Psine*2*R)
    Pnoise=Psine/(10**(SNRdB/10))
    vsigma=np.sqrt(Pnoise*R)
    tt = np.arange(start = 0, step = 1/fs, stop = nn/fs)
    _, xx_sine = mi_funcion( vmax = vmax, dc = 0, ff = ff, ph= ph, nn = nn, fs = fs, type_s = 'sine')
    _ ,xx_noise = mi_funcion( vmed = 0, vsigma = vsigma, nn = nn, fs = fs, type_s = 'noise')
    xx= xx_sine + xx_noise
    SNR_real = 10 * np.log10(Psine / np.var(xx_noise))
    print(f"SNR real: {SNR_real:.9f} dB")
    return tt, xx, xx_sine , xx_noise

def DFT(x):
    
    N=len(x)
    n = np.arange(N)
    k=n.reshape((N,1)) #Vector fila a Vector columna
    e=np.exp(-2j*np.pi*k*n/N)
    Xk=np.dot(x,e) #Producto Punto.
    
    return Xk 


N=1000
fs   = 100 #Resolucion espectral normalizada
deltaf=fs/N
ff = fs/8

t1, x = mi_funcion( vmax = np.sqrt(2), dc = 0, ff = ff, ph = 0, nn = N, fs = fs, type_s = 'sine')

ff2 = fs/8 - deltaf/2

t2, x2 = mi_funcion( vmax = np.sqrt(2), dc = 0, ff = ff2, ph = 0, nn = N, fs = fs, type_s = 'sine')

win_hamming = signal.windows.hamming(N)
win_flattop = signal.windows.flattop(N)

norma_hamming = len(win_hamming)/np.sum(win_hamming)
win_hamming = win_hamming * norma_hamming

norma_flattop = len(win_flattop)/np.sum(win_flattop)
win_flattop = win_flattop * norma_flattop

xdft = DFT(x)
xdft2_flattop = DFT(x2*win_flattop)
xdft2_hamming = DFT(x2*win_hamming)
xdft2 = DFT(x2)
freq= np.arange(start = 0, stop = fs, step = deltaf)

plt.figure()
plt.title("Tiempo")
plt.plot(t1,x, label= "Juguete" )
plt.plot(t2,x2, label= "Real" )
plt.xlim(0, 0.25)
plt.grid()
plt.legend()


plt.figure()
plt.title("Modulo Potencia")
plt.plot(freq, 10*np.log10(2*(np.abs(xdft)/len(xdft))**2),    label="Juguete", marker='x', linestyle='None',markersize = 10)
plt.plot(freq, 10*np.log10(2*(np.abs(xdft2)/len(xdft2))**2),  label="Real",    marker='o', linestyle='None')
plt.plot(freq, 10*np.log10(2*(np.abs(xdft2_hamming)/len(xdft2_hamming))**2), label="Hamming", marker='o', linestyle='None')
plt.plot(freq, 10*np.log10(2*(np.abs(xdft2_flattop)/len(xdft2_flattop))**2), label="Flattop", marker='o', linestyle='None')
plt.xlim(0, fs//2)
plt.grid()
plt.legend()

plt.figure()
plt.title("Modulo Tension")
plt.plot(freq, 2*(np.abs(xdft)/len(xdft)),             label="Juguete", marker='x', linestyle='None',markersize = 10)
plt.plot(freq, 2*(np.abs(xdft2)/len(xdft2)),           label="Real",    marker='o', linestyle='None')
plt.plot(freq, 2*(np.abs(xdft2_hamming)/len(xdft2_hamming)), label="Hamming", marker='o', linestyle='None')
plt.plot(freq, 2*(np.abs(xdft2_flattop)/len(xdft2_flattop)), label="Flattop", marker='o', linestyle='None')
plt.xlim(0, fs//2)
plt.grid()
plt.legend()





