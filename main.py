#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:50:47 2024

@author: sabrinaperrenoud
"""
#!/usr/bin/env python3


import matplotlib.pyplot as plt
import os
from natsort import natsorted
import numpy as np
import pandas as pd

import tools as tools
import classifier as classify
import fitting as fit




#%%

'''
File directory
    Directory format as folder with date of data collected
'''
directory = 'data/2024-01-11/'
output_directory = 'output/2024-01-11/'
datasetnames = classify.mac_natsorted(os.listdir(directory))
print('Available folders: ', datasetnames)

#%%
'''
Saving individual .csv files into crunched single .csv file
'''
crop = [4500,5500]

classify.save_file(directory, datasetnames, crop, bg=False)
#%%
'''
Loading in files from saved_data directory
    Combines all cropped datasets
'''
# crop_list, gaba, gaba_corr = tools.combine_files(directory)
direct = 'data/'
date = '2024-01-18'
directory = direct+date+'/'

bg, data = classify.load_file_simple(directory)


#%%

'''
Spectrum over time
    - Reshape data: tracking signal intensity for each frequency
    - Each list = 1 frequency over time (number of frames)

'''

datafile = data[0]
domain = [0,2425]
xdata = datafile[0]
ydata = datafile[1]
# ydata = np.array([y.tolist() for y in ydata])

print(len(ydata[0]))

fstart = 0
fend = 30



# bg_spectrum = tools.spectrum(bg[0], bg[1], frame_crop=[0,-1], start = fstart, end = fend, vmin=-100, vmax=-80, save=False, target_dir = 'output/')
# water_spectrum = tools.spectrum(water[0], water[1], frame_crop=[0,-1], start = fstart, end = fend, vmin=0, vmax=40, save=False, target_dir = 'output/')
# gaba_spectrum = tools.spectrum(xdata, ydata, domain, frame_crop=[0,-1], start = fstart, end = fend, vmin=-100, vmax=-80, save=False, target_dir = 'output/')
# gabacorr_spectrum = fit.spectrum(xdata, ydata, domain, frame_crop=[0,-1], start = fstart, end = fend, vmin=0, vmax=30, save=False, target_dir = 'output/')

# gabacorr_spectrum = fit.spectrum2(xdata, datafile[1][0], domain, save = True, target_dir = 'output/'+date+'/')


#%%
'''
Peak finder settings
'''
width = 10
height = -110
distance = 10
'''
Cycling multiple Lorentzian fit through data frames
    - multi-Lorentzian fit
'''


xdata = datafile[0][30:]
ydata = [d[30:] for d in datafile[1]]

n_frames = [1400, 1800]

amplitudes, frequencies, widths, fits = fit.cycle_fit(xdata, ydata, n_avg = 30, n_frames = n_frames, minfit=False, threshold=None, width=width, height=height, distance=distance, prominence=None, target_dir = 'output/'+date+'/', plot=False)


#%%
'''
Loading Saved fit_parameters
    -> combining data files if necesary
        via classify.combine)fir_parameters
'''

target_dir = 'output/'+date+'/'

classify.combine_fit_parameters(target_dir)

# amplitudes, frequencies, widths, fits = np.load(target_dir+'[1000, 1200]_fit_parameters.npy', allow_pickle=True)  

xframes, amplitudes, frequencies, widths, fits = np.load(target_dir+'fit_parameters.npy', allow_pickle=True)  

# print(amplitudes[3])
# nan=[np.nan] * 64
# amplitudes[3] =amplitudes[3]+ nan
# print(amplitudes[3])
# print(len(amplitudes[3]))
# np.save(target_dir+str([1000, 1200])+"_fit_parameters.npy", [amplitudes, frequencies, widths, fits])

#%%

'''
Beatnote characteristics over time
    - Fitted peak central frequency, x_0
    - Fitted peak amplitude, a
    - Fitted peak width, gam
'''

beatnote_frequencies = []
beatnote_amplitudes = []
beatnote_widths = []
for b, beatnote in enumerate(frequencies):
    t = xframes
    beatnote_frequencies.append([ t, beatnote ])
    beatnote_amplitudes.append([ t, amplitudes[b] ])
    beatnote_widths.append([ t, widths[b] ])
    
tools.xy_plot( beatnote_amplitudes, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Amplitude (dB)', title = 'Beatnote Amplitudes over {} frames'.format(str(len(t))), box = False, save = True, target_dir = 'output/'+date+'/')

tools.xy_plot( beatnote_widths, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Width (MHz)', title = 'Beatnote Widths over {} frames'.format(str(len(t))), box = False, save = True, target_dir = 'output/'+date+'/')

tools.xy_plot( beatnote_frequencies, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = 'Beatnote Frequencies over {} frames'.format(str(len(t))), box = False, save = True, target_dir = 'output/'+date+'/')

#%%
'''
Filtering out outliers from dataset
'''
# amplitudes2, frequencies2, widths2, frames2, frame_outliers = fit.remove_failed_fits(amplitudes, frequencies, widths, ydata, height)

# print(frame_outliers[1])
# fit.check_outliers(xdata, ydata, fits, 30, frame_outliers[1][50:60])

# tools.xy_plot([[t, frequencies2[0]]], type='beat_timeline', label_variable = None, aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = 'Filtered Beat Frequency 1 over {} frames'.format(n_frames), box = False, save = False, target_dir = 'output/')
# tools.xy_plot([[t, frequencies2[1]]], type='beat_timeline', label_variable = None, aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = 'Filtered Beat Frequency 2 over {} frames'.format(n_frames), box = False, save = False, target_dir = 'output/')

#%%
'''
Beat Note Selection
    - Derivative of each beat note frequency over time
    - Select high-order beat notes with derivate > threshold
'''
tools.xy_plot( beatnote_frequencies, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = 'Beatnote Frequencies over {} frames'.format(str(len(t))), box = False, save = True, target_dir = 'output/'+date+'/')

threshold = 0.05
select_data = frequencies
selected_times, selected_beats = fit.select_higher_beats(select_data, threshold = threshold)


#%%
'''
1D Histogram
    - Using beat note difference (derivative)
    - of Selected beat notes above threshold
'''
n_bins = 30

for h, hist_data in enumerate(selected_beats):
    n_points = len(hist_data)
    tools.xy_plot([hist_data,n_bins], type = 'histogram', aspect = 1.0, yerror = None, x_label = r'Beat Step $\Delta \nu$ (MHz)', y_label = 'n', title = '1D Histogram for Beatnote {}'.format(str(h+1)), box = False, save = True, target_dir = 'output/'+date+'/')



