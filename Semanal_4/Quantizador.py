#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal as sp
import time
from scipy.fft import fft, ifft

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

N=1000
fs   = N #Resolucion espectral normalizada
deltaf=fs/N
ff = deltaf*10

t, x = mi_funcion( vmax = np.sqrt(2), dc = 0, ff = ff, ph = 0, nn = N, fs = fs, type_s = 'sine')




VF=2
VFS=2*VF
B = 8

def Quant(VFS, x, B) :
    
    q = VFS/(2**B)
    SQ = np.round(x/q)*q
    SQ = np.clip(SQ,a_min=-q*(2**(B-1)) , a_max=q*(2**(B-1)-1))
    print(q)
    return SQ
# %%

SQ = Quant(VFS,x,B)
plt.plot(t,x)
plt.title("Senal y Cuantizada")
plt.minorticks_on()
plt.plot(t,SQ, drawstyle = 'steps-post')
plt.xlim([0, 2/ff])
plt.grid(which = 'major', linestyle = '-', linewidth = 0.8)
plt.grid(which = 'minor', linestyle = '-', linewidth = 0.8)

# %%
SE = x - SQ

plt.figure()
plt.title("Error")
plt.plot(t , SE, drawstyle = 'steps-post')
plt.minorticks_on()
plt.grid(which = 'major', linestyle = '-', linewidth = 0.8)
plt.grid(which = 'minor', linestyle = '-', linewidth = 0.8)

autocorr = np.correlate(SE, SE, mode='full')

print(autocorr)
# %%
# %%

freq= np.arange(start = 0, stop = fs, step = deltaf)
freq2=np.arange(start = 0, stop = fs, step = deltaf/2)
freq2 = freq2[:-1]
sqf = fft(SQ)

xft = fft(x)

plt.figure()
plt.title("FFT Se;al")
plt.plot(freq,20*np.log10(2*np.abs(sqf)/N), marker='x', label= 'quant')
plt.plot(freq,20*np.log10(2*np.abs(xft)/N), label= 'signal')
plt.grid()
plt.legend()
plt.xlim([0, fs//2])

# %%
# %%

autocorrfft = fft(autocorr/N)

plt.figure()
plt.title("FFT")
plt.plot(freq2,10*np.log10(2*np.abs(autocorrfft)/N), marker='x', label= 'error')
plt.plot(freq,20*np.log10(2*np.abs(xft)/N), label='signal')
plt.grid()
plt.legend()
plt.xlim([0, fs//2])


# %%
# %%

plt.figure()
plt.title("FFT error")
plt.plot(freq2,10*np.log10(2*np.abs(autocorrfft)/N), marker='x', label= 'error')
plt.grid()
plt.legend()
plt.xlim([0, fs//2])

# %%
#Funcion del generador de se;ales
# %%
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

# %%
# %%
SNR = 80
tt, xx ,  xx_sine , xx_noise = mi_generador_ruidoso( Psine = 1,R = 1 ,SNRdB=SNR, ff = ff, ph = 0, nn = N, fs = fs)
SQ2 = Quant(VFS,xx,B)
plt.figure()
plt.title("Senal ruidosa y Cuantizada")
plt.minorticks_on()
plt.plot(tt,xx)
plt.plot(tt,SQ2, drawstyle = 'steps-post')
plt.xlim([0, 2/ff])
plt.grid(which = 'major', linestyle = '-', linewidth = 0.8)
plt.grid(which = 'minor', linestyle = '-', linewidth = 0.8)
# %%

# %%
xxft = fft(xx)
sq2f= fft(SQ2)

se2= xx-SQ2


se2ff = fft(se2)

sn= fft(xx_noise)
# %%
# %%


plt.figure()
plt.title("FFT Se;al ruidosa")
plt.plot(freq,20*np.log10(2*np.abs(sq2f)/N), marker='x', label= 'quant')
plt.plot(freq,20*np.log10(2*np.abs(xxft)/N), label= 'signal')
plt.grid()
plt.legend()
plt.xlim([0, fs//2])

# %%
# %%


plt.figure()
plt.title("FFT ruidosa")
plt.plot(freq,20*np.log10(2*np.abs(se2ff)/N), marker='x', label= 'Ruido digital')
plt.plot(freq,20*np.log10(2*np.abs(sn)/N), marker='o', label= 'Ruido analogico')
plt.plot(freq,20*np.log10(2*np.abs(xxft)/N), label='signal')
plt.plot(freq,20*np.log10(2*np.abs(sq2f)/N), label='Cuantizada')
plt.grid()
plt.legend()
plt.xlim([0, fs//2])
# %%



plt.figure()
plt.title("FFT error")
plt.plot(freq2,10*np.log10(2*np.abs(se2ff)/N), marker='x', label= 'error')
plt.grid()
plt.legend()
plt.xlim([0, fs//2])
