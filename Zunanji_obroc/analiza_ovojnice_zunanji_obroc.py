import xlrd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib
import plotly
from plotly.graph_objs import Scatter, Layout
from scipy.signal import butter, filtfilt, hilbert, lfilter, sosfilt, sosfreqz
from scipy.fftpack import fft
import scipy.integrate
from scipy import signal


#Direct input
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams["figure.figsize"] = (6.2,3.5)
plt.rcParams.update(params)

wb1 = xlrd.open_workbook('Etalon_serija_kos_3.xlsx')
wb2 = xlrd.open_workbook('Napaka_zunanji_obroc_kos_1.xlsx')


sh3 = wb1.sheet_by_name(u'3600')
sh33 = wb2.sheet_by_name(u'3600')

time1_3600_etalon = sh3.col_values(0)  # time
acc1_3600_etalon = sh3.col_values(1)  # acc 1
acc2_3600_etalon = sh3.col_values(2)  # acc 2
acc3_3600_etalon = sh3.col_values(3)  # acc 3
mic1_3600_etalon = sh3.col_values(4)  # mic 1
mic2_3600_etalon = sh3.col_values(5)  # mic 2

time1_3600_napaka = sh33.col_values(0)  # time
acc1_3600_napaka = sh33.col_values(1)  # acc 1
acc2_3600_napaka = sh33.col_values(2)  # acc 2
acc3_3600_napaka = sh33.col_values(3)  # acc 2
mic1_3600_napaka = sh33.col_values(4)  # mic 1
mic2_3600_napaka = sh33.col_values(5)  # mic 2

dt = time1_3600_etalon[1]-time1_3600_etalon[0]
N = len(time1_3600_etalon)

#filter definicja butterworth

fs = 51200
lowcut=5000
highcut=7000
#filter definition

window = np.hanning(51200)


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_signal_napaka = sosfilt(sos, data)
    return filtered_signal_napaka

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_signal_etalon = sosfilt(sos, data)
    return filtered_signal_etalon

filtered_signal_napaka = window*butter_bandpass_filter(acc2_3600_napaka, lowcut, highcut, fs, order=4)
filtered_signal_etalon = window*butter_bandpass_filter(acc2_3600_etalon, lowcut, highcut, fs, order=4)

#operacija hilbert
hil_napaka = hilbert(filtered_signal_napaka)
hil_napaka_abs = np.abs(hil_napaka)

hil_etalon = hilbert(filtered_signal_etalon)
hil_etalon_abs = np.abs(hil_etalon)

#FFT
fft_hil_napaka = np.fft.rfft(hil_napaka_abs)*2/N
fft_napaka = np.fft.rfft(filtered_signal_napaka)*2/N


fft_hil_etalon = np.fft.rfft(hil_etalon_abs)*2/N
fft_etalon = np.fft.rfft(filtered_signal_etalon)*2/N


fr = np.fft.rfftfreq(N,d=dt)

##izračun frekvenc ležaja

RPM = 3600
fre = RPM/60
Bd = 3.97
Pd = 15.05
Nb = 7
theta = 13

#enačbe frekvenc ležaja

Bpfo = Nb/2*fre*(1-(Bd/Pd*np.cos(np.deg2rad(theta))))
Bpfi = Nb/2*fre*(1+(Bd/Pd*np.cos(np.deg2rad(theta))))
Bsf = Pd/(2*Bd)*fre*(1-(Bd/Pd)**2 * np.cos(np.deg2rad(theta))**2)
Ftf =fre/2*(1-(Bd/Pd*np.cos(np.deg2rad(theta))))


print(Bpfo)
print(Bpfi)
print(Bsf)
print(Ftf)

fig1, ax = plt.subplots()

plt.semilogy(fr,np.abs(fft_napaka),'r',linewidth=0.8,label='Puhalo z napako')
plt.semilogy(fr,np.abs(fft_etalon),'b',linewidth=0.8,label='Puhalo OK')


plt.ylabel(r'Pospešek $|a|$  $\mathrm{[m/s^2]}$')
plt.xlabel("Frekvenca $f$ [Hz]")
plt.grid( linestyle='--', linewidth=0.5)
plt.xlim(0,20000)
plt.legend(loc=4,frameon=True)

fig2, ax = plt.subplots()

plt.semilogy(fr,np.abs(fft_hil_napaka),'r',linewidth=0.8,label='Puhalo z napako')
plt.semilogy(fr,np.abs(fft_hil_etalon),'b',linewidth=0.8,label='Puhalo OK')
plt.ylabel(r'Pospešek $|a|$  $\mathrm{[m/s^2]}$')
#plt.ylabel(r'Zvočni tlak $|p|$  $\mathrm{[Pa]}$')
plt.xlabel("Frekvenca $f$ [Hz]")
plt.grid(True, which='both', linestyle='--', linewidth=0.2)
plt.xlim(0,1000)
#plt.ylim(0.00001,0.01)
plt.ylim(0.001,50)
plt.legend(loc=4)
plt.tight_layout()

#izriz fekvenc in harmonikov

# hline = ax.axvline(x=fre, ymin=0.50, ymax = 0.55, linewidth=0.5, color='black',ls='--')
# text = ax.text(fre, 0.55, "$f_{\mathrm{rot}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)
# hline = ax.axvline(x=2*fre, ymin=0.50, ymax = 0.55, linewidth=0.5, color='black',ls='--')
# text = ax.text(2*fre, 0.55, "2x$f_{\mathrm{rot}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)
# hline = ax.axvline(x=3*fre, ymin=0.8, ymax = 0.85, linewidth=0.5, color='black',ls='--')
# text = ax.text(3*fre, 0.85, "3xFr", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)
# hline = ax.axvline(x=4*fre, ymin=0.8, ymax = 0.85, linewidth=0.5, color='black',ls='--')
# text = ax.text(4*fre, 0.85, "4xFr", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)

# hline = ax.axvline(x=Bpfo, ymin=0.60, ymax = 0.65, linewidth=0.5, color='xkcd:black',ls='--')
# text = ax.text(Bpfo, 0.65, "$f_{\mathrm{bpfo}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='xkcd:black', fontsize=12)
# hline = ax.axvline(x=2*Bpfo, ymin=0.60, ymax = 0.65, linewidth=0.5, color='xkcd:black',ls='--')
# text = ax.text(2*Bpfo, 0.65, "2x$f_{\mathrm{bpfo}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='xkcd:black', fontsize=12)
# hline = ax.axvline(x=3*Bpfo, ymin=0.60, ymax = 0.65, linewidth=0.5, color='xkcd:black',ls='--')
# text = ax.text(3*Bpfo, 0.65, "3x$f_{\mathrm{bpfo}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='xkcd:black', fontsize=12)
# hline = ax.axvline(x=4*Bpfo, ymin=0.60, ymax = 0.65, linewidth=0.5, color='xkcd:black',ls='--')
# text = ax.text(4*Bpfo, 0.65, "4x$f_{\mathrm{bpfo}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='xkcd:black', fontsize=12)


# hline = ax.axvline(x=Bsf, ymin=0.75, ymax = 0.80, linewidth=0.5, color='xkcd:black',ls='--')
# text = ax.text(Bsf, 0.80, "$f_{\mathrm{bsf}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='xkcd:black', fontsize=12)
# hline = ax.axvline(x=2*Bsf, ymin=0.75, ymax = 0.80, linewidth=0.5, color='xkcd:black',ls='--')
# text = ax.text(2*Bsf, 0.80, "2x$f_{\mathrm{bsf}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='xkcd:black', fontsize=12)
# hline = ax.axvline(x=3*Bsf, ymin=0.75, ymax = 0.80, linewidth=0.5, color='xkcd:black',ls='--')
# text = ax.text(3*Bsf, 0.80, "3x$f_{\mathrm{bsf}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='xkcd:black', fontsize=12)
# hline = ax.axvline(x=4*Bsf, ymin=0.75, ymax = 0.80, linewidth=0.5, color='xkcd:black',ls='--')
# text = ax.text(4*Bsf, 0.80, "4x$f_{\mathrm{bsf}}$", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='xkcd:black', fontsize=12)

#primer za analizo balansiranja z integriranjem

# vel_etalon = 1000*scipy.integrate.cumtrapz(filtered_signal_etalon,time1_3600_etalon)
# vel_napaka = 1000*scipy.integrate.cumtrapz(filtered_signal_napaka,time1_3600_etalon)

# fft_vel_etalon = np.fft.rfft(vel_etalon)*2/N
# fft_vel_napaka = np.fft.rfft(vel_napaka)*2/N
# fr = np.fft.rfftfreq(N,d=dt)

# fig3, ax = plt.subplots()
# plt.ylabel(r'Hitrost $v$ [mm/s]')
# plt.xlabel(r'Frekvenca $f$ [Hz]')
# plt.grid( linestyle='--', linewidth=0.5)
# plt.xlim(0,6000)
# plt.ylim(0,4)
# plt.legend(loc=1,frameon=True)
# plt.tight_layout()

#izračun RMS vrednosti



lowcut_rms= 150
highcut_rms= 160

def butter_bandpass_filter(data, lowcut_rms, highcut_rms, fs, order=4):
    sos = butter_bandpass(lowcut_rms, highcut_rms, fs, order=order)
    filtered_signal_napaka_rms = sosfilt(sos, data)
    return filtered_signal_napaka_rms

def butter_bandpass_filter(data, lowcut_rms, highcut_rms, fs, order=4):
    sos = butter_bandpass(lowcut_rms, highcut_rms, fs, order=order)
    filtered_signal_etalon_rms = sosfilt(sos, data)
    return filtered_signal_etalon_rms

filtered_signal_napaka_rms = butter_bandpass_filter(hil_napaka_abs , lowcut_rms, highcut_rms, fs, order=4)
filtered_signal_etalon_rms = butter_bandpass_filter(hil_etalon_abs , lowcut_rms, highcut_rms, fs, order=4)

rms_napaka_acc=np.sqrt(np.mean(filtered_signal_napaka_rms**2))
rms_etalon_acc=np.sqrt(np.mean(filtered_signal_etalon_rms**2))
print('napaka rms = ',rms_napaka_acc)
print('etalon rms = ',rms_etalon_acc)

plt.show()
