#
# Copyright (C) 2018-2022 Pico Technology Ltd. See LICENSE file for terms.
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import pickle
import os
from natsort import natsorted


'''
PS5000A FUNCTIONS
'''

def sampling_interval(timebase):
    if timebase <=2:
        sampling_interval = (2**timebase) / 10**9
    else:
        sampling_interval = (timebase-2) / 125000000
    return sampling_interval


def dBu2V(dbu):
    #dbu = 0.775 * V
    V = dbu/0.775
    return V

def V2dBu(V):
    dBu = 0.775 * V
    return dBu

def plot_fft(x, y):
    time = x
    N = (len(y) - 1)
    T = time[1]-time[0]
    y = y
    xf = scipy.fftpack.fftfreq(N, T)[:N//2]
    yf = scipy.fftpack.fft(y)
    
    # yf = V2dBu(yf)
    yf = 20*np.log10(yf)
    
    # xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fig, ax = plt.subplots()
    # ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    ax.plot(xf, yf[:N//2])
    # plt.savefig('output/channelA_fft.png')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Intensity (dB)')
    plt.show()

def get_fft(x, data, date):
    n_frames = len(data)
    fft_signals=[]
    for s, signal in enumerate(data):
        time = x
        y = signal
        N = (len(y) - 1)
        T = time[1]-time[0]
        xf = scipy.fftpack.fftfreq(N, T)[:N//2]
        yf = scipy.fftpack.fft(y)
        
        # yf = V2dBu(yf)
        yf = 20*np.log10(yf)
        yf = yf[:N//2]
        fft_signals.append(yf)    
    
        
    np.save('output/'+date+'/data/fft_data_{}_nframes{}.npy'.format(date, n_frames), np.array([xf, fft_signals]), dtype=object)
    return fft_signals

def mac_natsorted(list):

    output = natsorted(list)
    if '.DS_Store' in output: output.remove('.DS_Store')
    return output

def load_pickled_data(date, filename, n_frames):
    directory = "output/"+date+"/data/"
    
    if filename is None:
        filename = 'raw_data_cumul_{}_nframes{}.txt'.format(date, n_frames)
    
    unpickled_data=[]
    with open(directory+filename, 'rb') as file:
        # reader = csv.reader(file)
        # dataopen2 = list(reader)
        for _ in range(n_frames):
            dataopen = pickle.load(file)
            unpickled_data.append(dataopen)
    return np.array(unpickled_data)



n_frames = 100
date = '2024-02-16'
directory = 'output/'+date+'/data/'


time = np.load(directory+'channelA_time.npy', allow_pickle = True)
data_cumul = load_pickled_data(date, None, n_frames)
data_full = np.load(directory+'raw_data_{}_nframes{}.txt'.format(date, n_frames), allow_pickle = True)

data = data_cumul

plt.plot(time, data[0])
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.show()

plot_fft(time, data[0])

#%%
'''
FFT Spectrum
'''

get_fft(time, data)
fft_datasets = np.load('output/'+date+'fft_data_{}_nframes{}.npy'.format(date, n_frames), allow_pickle = True)
