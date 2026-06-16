import pandas as pd
from scipy.io import loadmat, whosmat
import numpy as np
from scipy.signal import windows as win
import matplotlib.pyplot as plt
import scipy.signal as sp
from pytc2.sistemas_lineales import plot_plantilla ,group_delay, GroupDelay


def plot_iir_response(sos_iir, fs, iir_lbl = 'un_IIR'):

    w_rad  = np.append(np.logspace(-3, 0.8, 1000), np.logspace(0.9, 1.8, 1000) )
    w_rad  = np.append(w_rad, np.linspace(64, (fs/2), 1000, endpoint=True) ) / (fs/2) * np.pi
    
    _, h_iir = sp.freqz_sos(sos_iir, worN=w_rad)
    
    w = w_rad / np.pi * (fs/2)
    
    this_lbl = iir_lbl + ' {:d}'.format(sos_iir.shape[0]*2)
    
    plt.figure(1)
    
    plt.plot(w, 20*np.log10(np.abs(h_iir)+1e-12), label= this_lbl )

        
                
    plt.figure(2)
    
    phase_fir = np.angle(h_iir)
    
    plt.plot(w, phase_fir, label= this_lbl)    # Bode phase plot
    
    plt.figure(3)
    
    gd_fir = group_delay(w_rad, phase_fir)
    
    # Para órdenes grandes 
    plt.plot(w[gd_fir > 0], gd_fir[gd_fir>0], label=this_lbl )    # Bode phase plot
    
    return int(np.round(np.median(gd_fir[np.bitwise_and(w > 3, w < 20)])))



# --- MAT ---
fs = 1000
mat_struct = loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
t_ecg = np.arange(N) / fs
# Flatten the heartbeat patterns (same as you did for ecg_one_lead)
hb_1 = mat_struct['heartbeat_pattern1'].flatten()
hb_2 = mat_struct['heartbeat_pattern2'].flatten()

# Create their own time axes based on their actual length
t_hb1 = np.arange(len(hb_1)) / fs
t_hb2 = np.arange(len(hb_2)) / fs

# # Plot
# plt.figure()
# plt.plot(t_ecg[5000:12000], ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(t_hb1, hb_1)   # ✅ matching shapes

# plt.figure()
# plt.plot(t_hb2, hb_2)   # ✅ matching shapes




# ------------------- FILTRADO ----------------------------#

# filter design
ripple = 0.5 # dB
atenuacion = 20 # dB

ws1 = .001 #Hz
wp1 = 1 #Hz
wp2 = 35.0 #Hz
ws2 = 49 #Hz



sos_iir_cheby2 = sp.iirdesign(wp=[wp1-0.1,wp2+1], ws=[ws1,ws2], gpass=ripple, gstop=atenuacion, output='sos', analog=False, ftype='cheby2', fs = fs)
sos_iir_butter = sp.iirdesign(wp=[wp1-0.1,wp2+1], ws=[ws1,ws2], gpass=ripple, gstop=atenuacion, output='sos', analog=False, ftype='butter', fs = fs)

w_rad = np.concatenate( [ np.logspace(start = -2, stop = np.log10(2), num = 250 ),
                       np.linspace(start = 2.1, stop = 34, num = 250 ),
                       np.logspace(start = np.log10(34.1), stop = np.log10(52), num = 250 ),
                       np.linspace(start = 52.1, stop = fs/2, num = 250 )] )

w_rad, h_iir_cheby2 = sp.freqz_sos(sos_iir_cheby2, worN=w_rad, fs = fs)
w_rad, h_iir_butter = sp.freqz_sos(sos_iir_butter, worN=w_rad, fs = fs)

plt.figure()

plt.plot(w_rad, 20*np.log10(np.abs(h_iir_cheby2)+1e-12), label = 'IIR Cheby 2' )
plt.plot(w_rad, 20*np.log10(np.abs(h_iir_butter)+1e-12), label = 'IIR Butter' )

plt.title('Filtros')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.grid()
plt.axis([0, 90, -80, 5 ]);

axes_hdl = plt.gca()

plot_plantilla(filter_type = 'bandpass', fpass = (wp1, wp2), fstop =  (ws1, ws2), ripple = ripple, attenuation = atenuacion, fs = fs)
_ = axes_hdl.legend()

plt.show()
print(f"Orden del Filtro IIR Cheby 2: {sos_iir_cheby2.shape[0]*2}")
print(f"Orden del Filtro IIR Butter : {sos_iir_butter.shape[0]*2}")



# -------------------------------------------------
# Modulo y fase
# -------------------------------------------------

# plt.figure(figsize=(12,8))

# plt.subplot(2,1,1)
# plt.plot(w, 20*np.log10(np.abs(h)))
# plt.title("Respuesta en Frecuencia (Módulo)")
# plt.ylabel("Magnitud [dB]")
# plt.xlabel("Frecuencia [Hz]")
# plt.grid(True)

# plt.subplot(2,1,2)
# plt.plot(w, np.unwrap(np.angle(h)))
# plt.title("Respuesta en Frecuencia (Fase)")
# plt.ylabel("Fase [rad]")
# plt.xlabel("Frecuencia [Hz]")
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# -------------------------------------------------
# Retardo de grupo 1
# -------------------------------------------------

# gp = -np.diff(np.unwrap(np.angle(h))) / np.diff(w*2*np.pi)

# plt.figure(figsize=(8,4))
# plt.plot(w[:-1], gp)  # ✅ w recortado a (N-1,)
# plt.title("Retardo de Grupo")
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Retardo [muestras]")
# plt.xlim([0,60])
# plt.ylim([-5,5])
# plt.grid(True)
# plt.show()

# -------------------------------------------------
# Retardo de grupo 2
# -------------------------------------------------

# b, a = sp.sos2tf(sos_iir)
# w_gd, gd = sp.group_delay((b, a), fs=fs)

# plt.figure(figsize=(8,4))
# plt.plot(w_gd, gd)
# plt.title("Retardo de Grupo")
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Retardo [muestras]")
# plt.xlim([0,60])
# plt.grid(True)
# plt.show()

# -------------------------------------------------
# Respuesta al impulso
# -------------------------------------------------

# imp = np.zeros(1000)
# imp[0] = 1
# imp_resp = sp.sosfilt(sos_iir, imp)

# plt.figure(figsize=(8,4))
# plt.stem(imp_resp)
# plt.title("Respuesta al Impulso")
# plt.xlabel("n [muestras]")
# plt.ylabel("Amplitud")
# plt.grid(True)
# plt.show() 


ECG_F_cheby2 = sp.sosfiltfilt(sos_iir_cheby2, ecg_one_lead)
ECG_F_butter = sp.sosfiltfilt(sos_iir_butter, ecg_one_lead)




###################################
#%% Regiones de interés con ruido #
###################################
 

 
###################################
#%% Regiones de interés sin ruido #
###################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ECG_F_cheby2[zoom_region], label='cheby2')
    plt.plot(zoom_region, ECG_F_butter[zoom_region], label='butter')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering with IIR example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.grid()
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()