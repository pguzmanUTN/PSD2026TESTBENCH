import pandas as pd
from scipy.io import wavfile, loadmat, whosmat
import numpy as np
from scipy.signal import windows as win
import matplotlib.pyplot as plt
import scipy.signal as sp

def blackman_tukey(x, fs=1, M=None, window='blackman'):
    ##Autocorrelaccion mediante FFT porque si no mi computadora la pasa muy mal
    x = x.ravel().astype(float)
    N = len(x)
    if M is None:
        M = N // 5
    X = np.fft.fft(x)
    xcorr_full = np.fft.ifft(X * np.conj(X)).real / (N * fs)
    r = np.concatenate([xcorr_full[N-(M-1):], xcorr_full[:M]])
    ventana = win.get_window(window, len(r))
    r_windowed = r * ventana
    psd = np.abs(np.fft.fft(r_windowed, n=N))
    f = np.fft.fftfreq(N, d=1/fs)
    mitad = f >= 0
    f_half = f[mitad]
    psd_half = psd[mitad]
    psd_half[1:] *= 2
    return f_half, psd_half

def potencia_acumulada(f, psd):
    df = f[1] - f[0]
    P_acum = np.cumsum(psd) * df
    return P_acum

def plot_psd(signal, fs, titulo):
    signal = signal.ravel().astype(float)
    N = len(signal)
    f1, P1 = sp.welch(signal, fs, window='hann', nperseg=N//5)
    f2, P2 = sp.welch(signal, fs, window='hann')
    f3, P3 = blackman_tukey(signal, fs=fs)
    f4, P4 = sp.periodogram(signal, fs, window='hann')

    metodos = [
        (f1, P1, 'Welch nperseg=N/5'),
        (f2, P2, 'Welch default'),
        (f3, P3, 'Blackman-Tukey'),
        (f4, P4, 'Periodogram'),
    ]

    # --- PSD ---
    fig, ax = plt.subplots()
    for f, P, label in metodos:
        ax.semilogy(f, P, label=label)
    ax.set_title(titulo)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    ax.legend()
    ax.grid()
    plt.show()

    # --- Potencia acumulada comparativa ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Potencia acumulada - {titulo}')

    P_totales = []
    f_95s = []

    for f, P, label in metodos:
        P_acum = potencia_acumulada(f, P)
        P_total = P_acum[-1]
        f_95 = f[np.argmax(P_acum >= 0.95 * P_total)]
        P_totales.append(P_total)
        f_95s.append(f_95)

        # Izquierda: potencia acumulada normalizada
        axes[0].plot(f, P_acum / P_total * 100, label=label)

    axes[0].axhline(95, color='k', linestyle='--', label='95%')
    axes[0].set_xlabel('Frecuencia [Hz]')
    axes[0].set_ylabel('Potencia acumulada [%]')
    axes[0].set_title('Potencia acumulada normalizada')
    axes[0].legend()
    axes[0].grid()

    # Derecha: barras comparativas de P_total y BW al 95%
    labels = [m[2] for m in metodos]
    x = np.arange(len(labels))
    width = 0.35

    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, P_totales, width, label='P total [V²]')
    ax2.set_ylabel('Potencia total [V²]')
    ax2.set_title('Potencia total y BW al 95%')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    ax2.grid(axis='y')

    ax3 = ax2.twinx()
    bars2 = ax3.bar(x + width/2, f_95s, width, color='orange', label='BW 95% [Hz]')
    ax3.set_ylabel('Ancho de banda 95% [Hz]')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # --- Print resumen ---
    P_tiempo = np.mean(signal**2)
    print(f"\n[{titulo}]  P tiempo = {P_tiempo:.4f} V²")
    print(f"  {'Método':<20} {'P total [V²]':>15} {'BW 95% [Hz]':>12}")
    print(f"  {'-'*50}")
    for (_, _, label), P_tot, f95 in zip(metodos, P_totales, f_95s):
        print(f"  {label:<20} {P_tot:>15.8f} {f95:>12.3f}")

# --- CSVs ---
tecla1 = np.genfromtxt(r"TEK0000.CSV", delimiter=",", usecols=(3,4))
num1 = tecla1[20:, 1]
t = tecla1[20:, 0]

ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)
fs_ppg = 400

# --- WAVs ---
fs_silbido, silbido     = wavfile.read('silbido.wav')
fs_cucaracha, cucaracha = wavfile.read('la cucaracha.wav')
fs_prueba, prueba_psd   = wavfile.read('prueba psd.wav')

# --- MAT ---
fs_mat = 1000
mat_struct = loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
t_ecg = np.arange(N) / fs_mat
hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']


# --- PSD de cada señal ---
plot_psd(ecg_one_lead, fs_mat,    'PSD - ECG')
plot_psd(num1,         1/(t[2]-t[1]),     'PSD - TEK CSV')
plot_psd(ppg,          fs_ppg,    'PSD - PPG')
plot_psd(silbido,      fs_silbido,    'PSD - Silbido')
plot_psd(cucaracha,    fs_cucaracha,  'PSD - La Cucaracha')
plot_psd(prueba_psd,   fs_prueba,     'PSD - Prueba PSD')