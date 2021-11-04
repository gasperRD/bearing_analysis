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

wb1 = xlrd.open_workbook('Etalon_debalans.xlsx')
wb2 = xlrd.open_workbook('Napaka_debalans_kos_1.xlsx')


sh3 = wb1.sheet_by_name(u'full')
sh33 = wb2.sheet_by_name(u'full')

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
lowcut=1
highcut=24000
#filter definition



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

filtered_signal_napaka = butter_bandpass_filter(acc2_3600_napaka, lowcut, highcut, fs, order=4)
filtered_signal_etalon = butter_bandpass_filter(acc2_3600_etalon, lowcut, highcut, fs, order=4)

vel_etalon = 1000*scipy.integrate.cumtrapz(filtered_signal_etalon,time1_3600_etalon)
vel_napaka = 1000*scipy.integrate.cumtrapz(filtered_signal_napaka,time1_3600_etalon)

RPM = 19700

fre = RPM/60

#FFT

fft_vel_etalon = np.fft.rfft(vel_etalon)*2/N
fft_vel_napaka = np.fft.rfft(vel_napaka)*2/N
fr = np.fft.rfftfreq(N,d=dt)

fr = np.fft.rfftfreq(N,d=dt)

fig3, ax = plt.subplots()
plt.plot(fr[:-1],np.abs(fft_vel_napaka),'r',linewidth=0.8,label='Vzorec št. 1')
plt.plot(fr[:-1],np.abs(fft_vel_etalon),'b',linewidth=0.8,label='Vzorec OK')


#plt.plot(fr,np.abs(fft_vel),'k',linewidth=0.8,label='int')

plt.ylabel(r'Hitrost $v$ [mm/s]')
plt.xlabel(r'Frekvenca $f$ [Hz]')

plt.grid( linestyle='--', linewidth=0.5)

plt.xlim(0,6000)
plt.ylim(0,4)
plt.legend(loc=1,frameon=True)
plt.tight_layout()

#vrtilna frekvenca
hline = ax.axvline(x=fre, ymin=0.8, ymax = 0.85, linewidth=0.5, color='black',ls='--')
text = ax.text(fre, 0.85, "Fr", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)
hline = ax.axvline(x=2*fre, ymin=0.8, ymax = 0.85, linewidth=0.5, color='black',ls='--')
text = ax.text(2*fre, 0.85, "2xFr", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)
hline = ax.axvline(x=3*fre, ymin=0.8, ymax = 0.85, linewidth=0.5, color='black',ls='--')
text = ax.text(3*fre, 0.85, "3xFr", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)
hline = ax.axvline(x=4*fre, ymin=0.8, ymax = 0.85, linewidth=0.5, color='black',ls='--')
text = ax.text(4*fre, 0.85, "4xFr", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)
hline = ax.axvline(x=5*fre, ymin=0.8, ymax = 0.85, linewidth=0.5, color='black',ls='--')
text = ax.text(5*fre, 0.85, "5xFr", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)
hline = ax.axvline(x=6*fre, ymin=0.8, ymax = 0.85, linewidth=0.5, color='black',ls='--')
text = ax.text(6*fre, 0.85, "6xFr", rotation=90,verticalalignment='bottom', horizontalalignment='center',transform=ax.get_xaxis_transform(),color='black', fontsize=12)


lowcut_rms= 20
highcut_rms= 700

def butter_bandpass_filter(data, lowcut_rms, highcut_rms, fs, order=4):
    sos = butter_bandpass(lowcut_rms, highcut_rms, fs, order=order)
    filtered_signal_napaka_rms = sosfilt(sos, data)
    return filtered_signal_napaka_rms

def butter_bandpass_filter(data, lowcut_rms, highcut_rms, fs, order=4):
    sos = butter_bandpass(lowcut_rms, highcut_rms, fs, order=order)
    filtered_signal_etalon_rms = sosfilt(sos, data)
    return filtered_signal_etalon_rms

filtered_signal_napaka_rms = butter_bandpass_filter(vel_napaka, lowcut_rms, highcut_rms, fs, order=4)
filtered_signal_etalon_rms = butter_bandpass_filter(vel_etalon, lowcut_rms, highcut_rms, fs, order=4)

rms_napaka_acc=np.sqrt(np.mean(filtered_signal_napaka_rms**2))
rms_etalon_acc=np.sqrt(np.mean(filtered_signal_etalon_rms**2))
print('napaka rms = ',rms_napaka_acc)
print('etalon rms = ',rms_etalon_acc)

plt.show()
