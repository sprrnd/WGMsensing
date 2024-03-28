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

class ResonancePipeline():
    
    def __init__(self, directory, folder, target_dir):
        self.directory = directory
        self.folder = folder
        self.target_dir = target_dir
        
    def load_data(self, type='mat', mac = True):
        if type == 'mat':
            self.data_folder = classify.load_mat_file(self.directory+self.folder+'/', mac=mac)
            self.data_dictionary = classify.process_dict(self.data_folder)
        if type == 'csv':
            self.data_dictionary = classify.load_csv_file(self.directory+self.folder+'/', mac=mac)
        if type == 'txt':
            self.data_dictionary = classify.load_txt_file(self.directory+self.folder+'/', skiprows=14, mac=mac)
            
    def plot_data(self, x, y):

        tools.xy_plot([x, y], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Frame {}'.format(str(index)), box = False, save = False, target_dir = self.target_dir)

    def classify_data(self, datafile):
        self.data_parameters = {}
        
        self.data_parameters['time'] = datafile[:, 0]
        self.data_parameters['resonance'] = datafile[:, 1]
        self.data_parameters['fwhm'] = datafile[:, 2]
        
        tools.xy_plot([[self.data_parameters['time'], self.data_parameters['resonance']]], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Time (s)', y_label = r'\lambda (nm)', title = 'Wavelength Raw Data', box = False, save = True, target_dir = self.target_dir)
        tools.xy_plot([[self.data_parameters['time'], self.data_parameters['fwhm']]], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Time (s)', y_label = 'FWHM (nm)', title = 'FWHM Raw Data', box = False, save = True, target_dir = self.target_dir)

        return self.data_parameters
    
    def remove_errors(self, x, y_name, error_range):
        y = self.data_parameters[str(y_name)]
        
        y_corr=[]
        for i, j in enumerate(y):
            if i>=min(error_range) and i<=max(error_range):
                y_corr.append(np.nan)
            else:
                y_corr.append(j)    
                
        self.data_parameters[str(y_name)+'_cut'] = y_corr
        
        tools.xy_plot([[x, y_corr]], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Time (s)', y_label = str(y_name), title = str(y_name)+' corrected for errors', box = False, save = True, target_dir = self.target_dir)

        return y_corr
    
    def correct_background(self, data_parameter, window_size):
        x = self.data_parameters['time']
        y = self.data_parameters[str(data_parameter)]
        
        moving_averages, y_corr = rem.background_correction(x, y, window_size)
        x_corr = np.linspace(min(x), max(x), len(y_corr))
        
        labels = ['Raw data', 'Background MA', 'Corrected data']
        tools.xy_plot([[x, y],[x_corr, moving_averages], [x_corr, y_corr]], fit = None, label_variable = labels, aspect = 0.5, yerror = None, x_label = 'Time (s)', y_label = 'Level', title = 'Background Correction', box = False, save = True, target_dir = self.target_dir)
        tools.xy_plot([[x_corr, y_corr]], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Time (s)', y_label = 'Wavelength (nm)', title = 'Background Corrected Data', box = False, save = True, target_dir = self.target_dir)
        
        self.data_parameters[str(data_parameter)+'_corr'] = y_corr
        self.data_parameters['time_corr'] = x_corr
        
        return y_corr

    
    def find_signal(self, x, y, type, interval, window_size):
        datax = x[interval[0] : interval[1]]
        datay = y[interval[0] : interval[1]]

        #if type == 'steps':
         #   maxima, minima, step_indices, steps, steps_excluded, delta_lambda, convolutions = fit.step_finder3(interval, datax, datay, window_size = window_size, type = 'step', minfit = True, distance = 50, width = 20, height = None, dev = 3, plot = True, target_dir = self.target_dir)
        #if type == 'spikes':
         #   maxima, minima, step_indices, steps, steps_excluded, delta_lambda, convolutions = fit.spike_finder(interval, datax, datay, window_size = window_size, type = 'spike', minfit = True, distance = 10, width = 3, height = None, dev = 3, plot = True, target_dir = self.target_dir)
        
        maxima, minima, step_indices, steps, steps_excluded, delta_lambda, convolutions = rem.signal_finder(interval, datax, datay, window_size = window_size, type = type, minfit = True, distance = 10, width = 3, height = None, dev = 3, plot = True, target_dir = self.target_dir)
        
    def load_signal_parameters(self, filename):
        #if mac is True:
         #   filenames =classify.mac_natsorted(os.listdir(self.target_dir))
        #else:
         #   filenames = os.listdir(self.target_dir)
        #filenames = [f for f in filenames if 'fit_parameters' in f and '._' not in f]
        #print('Fit parameters available to load: ', filenames)
        
        self.signal_parameters = {}
                       
        maxima, minima, step_indices, steps, steps_excluded, delta_lambda, convolutions = np.load(self.target_dir+filename, allow_pickle=True)

        self.signal_parameters['maxima'] = maxima
        self.signal_parameters['minima'] = minima
        self.signal_parameters['step_indices'] = step_indices
        self.signal_parameters['steps'] = steps
        self.signal_parameters['steps_excluded'] = steps_excluded
        self.signal_parameters['delta_lambda'] = delta_lambda
        self.signal_parameters['convolutions'] = convolutions
                     
                       
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
        
        self.data_filtered = rem.filter_outliers(data, sigma)
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
        self.selected_times, self.selected_beats = rem.select_higher_beatnotes(x, beatnote_frequencies, beatnote_index, sigma, self.target_dir)
        
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