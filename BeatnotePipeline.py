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
import scipy
from scipy import stats
from scipy import signal
from scipy.signal import correlate

import tools as tools
import classifier as classify
import fitting as fit

class BeatnotePipeline():
    
    def __init__(self, directory, folder, target_dir):
        self.directory = directory
        self.folder = folder
        self.target_dir = target_dir
        
    def load_data(self, type='mat'):
        if type == 'mat':
            self.data_folder = classify.load_mat_file(self.directory+self.folder+'/')
            self.data_dictionary = classify.process_dict(self.data_folder)
        if type == 'csv':
            self.data_dictionary = classify.load_csv_file(self.directory+self.folder+'/')
        
    def plot_data(self, datafile, index=0, fit = False):
        x = datafile[0]
        x = [i/10**6 for i in x]
        # y = [i/10**6 for i in x]      
        y = datafile[1]
        
        if fit is True:
            fit_i = self.fits[0][index]
            tools.xy_plot([[x, y[index]]], fit = [fit_i], label_variable = None, aspect = 0.5, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Frame {}'.format(str(index)), box = False, save = False, target_dir = self.target_dir)
            
        else:

            tools.xy_plot([[x, y[index]]], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Frame {}'.format(str(index)), box = False, save = False, target_dir = self.target_dir)
        
    def plot_spectrum(self, xdata, ydata, vmin, vmax, crop=None):
        xdata = [d/10**6 for d in xdata]
        self.spectrum = fit.spectrum(xdata, ydata, vmin=vmin, vmax=vmax, crop=crop, save=True, target_dir = self.target_dir)

        
    def correct_file(self, background_filename, filename):
        
        if background_filename is not None:
            background = self.data_dictionary[str(background_filename)]
            data = self.data_dictionary[str(filename)]

            data_corr = []
            bg_avg = tools.background_avg(background)
            for i, arr in enumerate(data[1]):
                data_corr.append(np.array(arr) - np.array(bg_avg))

            self.data_dictionary[str(filename)+'_corr'] = [frequency, data_corr]
        else:
            print('No background file available')
            data_corr = None

        return data_corr
    
        
    def process_data(self, datafile, crop, n_avg, width, height, distance, n_frames, plot=False):
        '''
        Cycling multiple Lorentzian fit through data frames
            - multi-Lorentzian fit
        '''
        xdata = datafile[0][crop[0] : crop[1]]
        ydata = [d[crop[0] : crop[1]] for d in datafile[1]]
        
        if height is None:
            floor = np.average(ydata[0])
            locmax = max(ydata[0][:10])
            locmin = min(ydata[0][:10])
            h = abs(locmax - locmin)
            height = 2*h + floor
            print(height)
        
        if distance is None:
            distance = width *10**6
            
        self.height = height
        self.n_frames = n_frames

        amplitudes, frequencies, widths, fits = fit.cycle_fit(xdata, ydata, n_avg, self.n_frames, minfit=False, threshold=None, width=width, height=height, distance=distance, prominence=None, target_dir = self.target_dir, plot=plot)

    def combine_saved_parameters(self):
        '''
        Loading Saved fit_parameters
            -> combining data files if necesary
                via classify.combine)fir_parameters
        '''
    
        classify.combine_fit_parameters(target_dir = self.target_dir)

    def load_fit_parameters(self):
        filenames =classify.mac_natsorted(os.listdir(self.target_dir))
        filenames = [f for f in filenames if 'fit_parameters' in f and '._' not in f]
        print('Fit parameters available to load: ', filenames)
        
        fit_params = {}
        for f in filenames:
            fit_params[f] = np.load(self.target_dir+f, allow_pickle=True)
        return fit_params
    
    def plot_timelines(self, fit_params, variable_type=None):
        '''
        Plot list variable y against single array x
        '''
        self.frames, self.amplitudes, self.frequencies, self.widths, self.fits = fit_params
        
        x = self.frames
        frame_no = len(self.frames)
        beat_no = len(self.frequencies)
        
        if variable_type == 'frequency':
            frequencies_mhz = [list(np.array(i)*10**-6) for i in self.frequencies]
            tools.xy_plot( [x, frequencies_mhz], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Beat Frequency (MHz)', title = 'Beatnote Frequency over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)
        if variable_type == 'amplitude':
            tools.xy_plot( [x, self.amplitudes], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Amplitude (dB)', title = 'Beatnote Amplitude over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)
        if variable_type == 'width':
            widths_mhz = [list(np.array(i)*10**-6) for i in self.widths]
            tools.xy_plot( [x, self.widths], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Width (MHz)', title = 'Beatnote Width over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)
        if variable_type == None:
            frequencies_mhz = [list(np.array(i)*10**-6) for i in self.frequencies]
            widths_mhz = [list(np.array(i)*10**-6) for i in self.widths]
            tools.xy_plot( [x, frequencies_mhz], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Beat Frequency (MHz)', title = 'Beatnote Frequency over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)            
            tools.xy_plot( [x, self.amplitudes], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Amplitude (dB)', title = 'Beatnote Amplitude over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)
            tools.xy_plot( [x, self.widths_mhz], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Width (MHz)', title = 'Beatnote Width over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)

        
    def filter_outliers(self, data, sigma = 3):
        
        self.data_filtered = fit.filter_outliers(data, sigma)
        tools.xy_plot( self.data_filtered, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(data))], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Filtered Variable', title = 'Filtered Beatnote Variable', box = False, save = True, target_dir = self.target_dir)
        
        #self.amplitudes_filtered = fit.filter_outliers(self.amplitudes, sigma)
        #tools.xy_plot( self.amplitudes_filtered, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(self.amplitudes))], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Amplitude (dB)', title = 'Filtered Beatnote Amplitudes over {} frames'.format(str(len(self.t))), box = False, save = True, target_dir = self.target_dir)
    
        #self.widths_filtered = fit.filter_outliers(self.widths, sigma)
        #tools.xy_plot( self.widths_filtered, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(self.widths))], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Width (MHz)', title = 'Filtered Beatnote Widths over {} frames'.format(str(len(self.t))), box = False, save = True, target_dir = self.target_dir)
        
        return self.data_filtered
        
        
    def get_beatnote_stats(self, x, beatnote_frequencies, beatnote_index=None, sigma=3, n_bins=30):
        '''
        Beat Note Selection
            - Derivative of each beat note frequency over time
            - Select high-order beat notes with derivate > threshold
        '''
        # tools.xy_plot(select_data, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(self.frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = 'Beatnote Frequencies over {} frames'.format(str(len(self.t))), box = False, save = True, target_dir = self.target_dir)
        self.selected_times, self.selected_beats = fit.select_higher_beatnotes(x, beatnote_frequencies, beatnote_index, sigma, self.target_dir)
        
        '''
        1D Histogram
            - Using beat note difference (derivative)
            - of Selected beat notes above threshold
        '''
        self.n_bins = n_bins
        
        if beatnote_index is None:
            for h, hist_data in enumerate(self.selected_beats):
                tools.xy_plot([hist_data,n_bins], type = 'histogram', aspect = 1.0, yerror = None, x_label = r'Beat Step $\Delta \nu$ (MHz)', y_label = 'n', title = '1D Histogram for Beatnote {}'.format(str(h+1)), box = False, save = True, target_dir = self.target_dir)
        else:
            tools.xy_plot([self.selected_beats, n_bins], type = 'histogram', aspect = 1.0, yerror = None, x_label = r'Beat Step $\Delta \nu$ (MHz)', y_label = 'n', title = '1D Histogram for Beatnote {}'.format(str(beatnote_index)), box = False, save = True, target_dir = self.target_dir)

        self.hist_data = self.selected_beats