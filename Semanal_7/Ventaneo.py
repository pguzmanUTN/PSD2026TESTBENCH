#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal as sp
import time
from scipy.fft import fft, ifft
import scipy.signal as signal
from scipy.stats import gaussian_kde

#Funcion del generador de se;ales
def  mi_funcion ( vmax = 1, dc = 0, ff = 1, ph = 0, nn = 100, fs = 50, type_s = 'sine', duty = 0.5, vmed=5e-3, vsigma=0.5e-3) :
    ff=np.array(ff)
    ff=ff.reshape((-1,1)) #Vector fila a Vector columna
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
fs   = 2*np.pi #Resolucion espectral normalizada
deltaf=fs/N
fr = np.random.uniform(-2,2,200)
ff = fs/4+fr*deltaf

tt1, x3d, _, _ = mi_generador_ruidoso( Psine = 1,R = 1 ,SNRdB=3, ff = ff, ph = 0, nn = N, fs = fs)
tt2, x10d, _, _ = mi_generador_ruidoso( Psine = 1,R = 1 ,SNRdB=10, ff = ff, ph = 0, nn = N, fs = fs)

win_hamming = signal.windows.hamming(N)
win_flattop = signal.windows.flattop(N)
win_blackman = signal.windows.blackmanharris(N)


norma_hamming = len(win_hamming)/np.sum(win_hamming)
win_hamming = win_hamming * norma_hamming

norma_flattop = len(win_flattop)/np.sum(win_flattop)
win_flattop = win_flattop * norma_flattop

norma_blackman = len(win_blackman)/np.sum(win_blackman)
win_blackman = win_blackman * norma_blackman




freq = np.arange(start = 0, step = deltaf ,stop = fs)


xdft_rect     = np.fft.fft(x10d,axis=1)
xdft_hamming  = np.fft.fft(x10d * win_hamming,axis=1)
xdft_flattop  = np.fft.fft(x10d * win_flattop,axis=1)
xdft_blackman = np.fft.fft(x10d * win_blackman,axis=1)

print("Sesgo")
a0 = np.sqrt(2)/2
a_rect = np.abs(xdft_rect[:, N//4])/N
a_hamming = np.abs(xdft_hamming[:, N//4])/N
a_flattop = np.abs(xdft_flattop[:, N//4])/N
a_blackman = np.abs(xdft_blackman[:, N//4])/N
    
var_rect = np.var(a_rect) 
var_hamming = np.var(a_hamming) 
var_flattop = np.var(a_flattop) 
var_blackman = np.var(a_blackman) 



Sa_rect =  a0 - np.mean(a_rect) 
Sa_hamming = a0 - np.mean(a_hamming) 
Sa_flattop = a0 - np.mean(a_flattop)
Sa_blackman = a0 - np.mean(a_blackman)

nombres = ['Rectangular', 'Hamming', 'Flat-top', 'Blackman-Harris']
sesgos  = [Sa_rect, Sa_hamming, Sa_flattop, Sa_blackman]
varianzas = [var_rect, var_hamming, var_flattop, var_blackman]

print("=" * 52)
print(f"{'Ventana':<18} {'Sesgo':>14} {'Varianza':>14}")
print("=" * 52)
for n, s, v in zip(nombres, sesgos, varianzas):
    print(f"{n:<18} {s:>+14.6f} {v:>14.3e}")
print("=" * 52)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Histogramas con densidad — SNR 10 dB", fontsize=14)

datos = [
    (a_rect,     'Rectangular',     'steelblue'),
    (a_hamming,  'Hamming',         'seagreen'),
    (a_flattop,  'Flat-top',        'darkorange'),
    (a_blackman, 'Blackman-Harris', 'mediumpurple'),
]

for ax, (a, nombre, color) in zip(axes.flat, datos):
    ax.hist(a, bins=30, color=color, alpha=0.35, edgecolor='white',
            linewidth=0.5, density=True)

    kde = gaussian_kde(a)
    x_kde = np.linspace(a.min(), a.max(), 300)
    ax.plot(x_kde, kde(x_kde), color=color, linewidth=2.5)

    ax.axvline(a0,         color='red',   linestyle='--', linewidth=1.5, label=f'a₀ = {a0:.4f}')
    ax.axvline(np.mean(a), color='black', linestyle='-',  linewidth=1.5, label=f'media = {np.mean(a):.4f}')

    ax.set_title(nombre)
    ax.set_xlabel("Amplitud estimada")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_title("Densidad de amplitud estimada — todas las ventanas", fontsize=13)

for a, nombre, color in datos:
    ax.hist(a, bins=30, color=color, alpha=0.2, edgecolor='none', density=True)

    kde = gaussian_kde(a)
    x_kde = np.linspace(a.min(), a.max(), 300)
    ax.plot(x_kde, kde(x_kde), color=color, linewidth=2.5, label=nombre)
    ax.axvline(np.mean(a), color=color, linestyle=':', linewidth=1.2)

ax.axvline(a0, color='red', linestyle='--', linewidth=1.8, label=f'a₀ verdadero = {a0:.4f}')

ax.set_xlabel("Amplitud estimada")
ax.set_ylabel("Densidad")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure()
plt.title("10 dB")
plt.xlabel("Radianes")
plt.plot( 2*(np.abs(xdft_rect[0])/len(xdft_rect[0])), label = 'rectangular',marker='o',markersize=8)
plt.plot( 2*(np.abs(xdft_flattop[0])/len(xdft_flattop[0])), label = 'flattop',marker='8',markersize=6)
plt.plot( 2*(np.abs(xdft_hamming[0])/len(xdft_hamming[0])), label = 'hamming',marker='D',markersize=4)
plt.plot( 2*(np.abs(xdft_blackman[0])/len(xdft_blackman[0])), label = 'blackman',marker='x',markersize=2)
plt.xlim(N/4 - 20, N/4 + 20)
plt.grid()
plt.legend()

