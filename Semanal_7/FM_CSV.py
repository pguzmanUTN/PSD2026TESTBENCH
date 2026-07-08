
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:57:19 2018

"""

# Traemos la libreria VISA
from scipy import signal
# Traemos matplotlib para poder graficar
import matplotlib.pyplot as plt
# Agreamos el path de las librerias
import numpy as np


def plot_spec(signal,fs,tipo="full",win="rect"):
    #espec=20*np.log10(np.abs(np.fft.fft(signal))/len(signal))
    espec=(np.abs(np.fft.fft(signal))/len(signal))
    espec=np.fft.fftshift(espec)
    fcia=np.linspace(-fs/2,fs/2,len(signal))
    if tipo=="half":
        espec=espec[len(signal)//2:]
        fcia=fcia[len(signal)//2:]
    plt.figure()
    plt.title('Espectro de la señal')
    #plt.ylim(np.max(espec)-75,np.max(espec)+5)
    plt.plot(fcia,espec)
    plt.show()

def estimate_freq_from_time_signal(t, x):
    """
    Estima la frecuencia de una señal senoide usando cruces por cero.
    t: vector de tiempos [s]
    x: vector de señal
    Devuelve frecuencia en Hz
    """
    # Eliminar componente DC
    x = x - np.mean(x)
    
    # Detectar cruces por cero (de signo)
    signs = np.sign(x)
    zero_crossings = np.where(np.diff(signs) != 0)[0]
    
    if len(zero_crossings) < 2:
        raise ValueError("No se detectaron suficientes cruces por cero.")
    
    # Calcular los tiempos de cruce por cero
    t_cross = t[zero_crossings]
    
    # Diferencias entre tiempos consecutivos (mitad de período)
    dt = np.diff(t_cross)
    
    # Frecuencia promedio (recordá que dos cruces = medio período)
    f_est = 1.0 / (2 * dt)
    return np.mean(f_est) , np.std(f_est,ddof=1) , len(f_est)

data1 = np.genfromtxt("FM.csv",delimiter=",",usecols=(0,1))
t = data1[0: , 0]
ch1 = data1[0: , 1] #Tension de entrada


# Generamos un operador y pedimos el valor RMS actual
#operador_1 = operador.Operador_osciloscopio(MiOsciloscopio,"Workbench_I")
fs = 1/(t[2]-t[1])
#N=4000
#fbt=10e-3
#div=10
#fs=N/(fbt*div)
#t1=np.linspace(0,N/fs,N)
#print("ts = {} s".format(t1[1]))
#print("f_Nys= {} Hz ".format(fs/2))
#print("fs",fs)


print("Frecuencia de Muestreo",fs)
plot_spec(ch1,fs)
#val_RMS = operador_1.medir_Vrms(canal = 1, VERBOSE = False)

#print('Vrms = %0.5f'%val_RMS)

ch1_shift=ch1*np.exp(1j*2*np.pi*1e6*t)

plot_spec(ch1_shift,fs)


filtro=signal.firwin(1001,350e3/fs)
plt.figure()
plt.title("respuesta temporal")
plt.plot(filtro) 
plot_spec(filtro,fs)

fm_filtrada=np.convolve(ch1_shift,filtro,mode="same")
plot_spec(fm_filtrada,fs)

plt.figure()
plt.plot(t,fm_filtrada.real,t,fm_filtrada.imag)
plt.show()

### demodulacion en cuadratura ####


def FM_demod (señal,fs):
    return np.angle(señal[:-1]*np.conjugate(señal[1:]))*fs/(2*np.pi)

mensaje=FM_demod(fm_filtrada,fs)

plt.figure()
plt.plot(t[:len(mensaje)],mensaje)
plt.show()

# Frecuencia de corte del filtro paso bajo (Hz)
fc = 5e3  # Sabemos que esta modulada por un tono entonces podemos poner una frecuencia  bien baja de corte
#del grafico se ve que tiene una frecuencia de aproximadamente 5kHz
# Orden del filtro
orden = 1001
# Normalizamos la frecuencia de corte
Wn = fc / (fs/2)

# Diseñamos un filtro FIR
lp_filtro = signal.firwin(orden, Wn)

# Filtramos la señal demodulada
mensaje_filtrado = np.convolve(mensaje, lp_filtro, mode='same')

# Graficamos
plt.figure()
plt.plot(t[:len(mensaje_filtrado)], mensaje_filtrado)
plt.title("Señal FM demodulada y filtrada")
plt.show()

# -----------------------------------------------
# Cálculo de desviación de frecuencia por semiciclo
# -----------------------------------------------

# Detectar cruces por cero para separar semiciclos
mensaje_filtrado = mensaje_filtrado.astype(float)  # asegurar tipo float
cruces = np.where(np.diff(np.sign(mensaje_filtrado)))[0]  # índices de cruce por cero

amplitudes_semiciclo = []

for i in range(len(cruces)-1):
    inicio = cruces[i]
    fin = cruces[i+1]
    # Tomamos el valor máximo absoluto en el semiciclo
    amp = np.max(np.abs(mensaje_filtrado[inicio:fin]))
    amplitudes_semiciclo.append(amp)

amplitudes_semiciclo = np.array(amplitudes_semiciclo)

promedio_amp = np.mean(amplitudes_semiciclo)
desv_std_amp = np.std(amplitudes_semiciclo,ddof=1)
u_amp = desv_std_amp/np.sqrt(len(amplitudes_semiciclo))

frecuencia_mensaje , desvio_f_mensaje , n_f_mensaje = estimate_freq_from_time_signal(t[:len(mensaje_filtrado)], mensaje_filtrado)
u_f_mensaje = desvio_f_mensaje/np.sqrt(n_f_mensaje)

betha = promedio_amp/frecuencia_mensaje
u_betha = betha*np.sqrt((u_amp/promedio_amp)**2+(u_f_mensaje/frecuencia_mensaje)**2)

print("Desviacion de frecuencia maxima promedio:", promedio_amp,"Hz")
print("Desviación estándar del desvio de frecuencia maxima promedio:", desv_std_amp , "Hz")
print("Incertidumbre sin expandir:", u_amp,"Hz")
print("\n")
print("Frecuencia del mensaje:", frecuencia_mensaje,"Hz")
print("Desviación estándar de frecuencia del mensaje:", desvio_f_mensaje , "Hz")
print("Incertidumbre sin expandir:", u_f_mensaje ,"Hz")
print("\n")
print("Indice de modulacion:", betha)
print("Incertidumbre del indice de modulacion sin expandir:", u_betha)



