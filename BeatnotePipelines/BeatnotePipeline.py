#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import os
import numpy as np

import tools as tools
import classifier as classify
import BeatnoteMethods as method

'''
BEATNOE PIPELINE

Data analysis pipeline for BeatnoteWorkbook
Parent class calling on BeatnoteMethods functions
'''

class BeatnotePipeline():
    
    def __init__(self, directory, folder, target_dir):
        '''
        Function: Initiates class by defining input and output directories
        Input:
            directory =  general folder where data is stored ('data/)
            folder = specific folder to load data from, usually date ('2024-')
            target_dir = output folder to save files, ('output/'+date)
        '''
        self.directory = directory
        self.folder = folder
        self.target_dir = target_dir
        
    def load_data(self, type='mat', mac=True):
        '''
        Function: Loads all data in folder corresponding to set data type
        Input:
            type = 'mat'/'csv/'txt
            mac = True/False
        Output:
            data_dictionary = stores all data under associated filenames
        '''
        if type == 'mat':
            self.data_folder = classify.load_mat_file(self.directory+self.folder+'/', mac=mac)
            self.data_dictionary = method.process_dict(self.data_folder)
        if type == 'csv':
            self.data_dictionary = classify.load_csv_file(self.directory+self.folder+'/', mac=mac)
        if type == 'txt':
            self.data_dictionary = classify.load_txt_file(self.directory+self.folder+'/', mac=mac)
        
    def plot_data(self, datafile, index=0):
        '''
        Function: Plots single data frame from datafile
        Input:
            datafile = specific dataset to plot
            index = frame index
        Output:
            single data frame plot
        '''
        x = datafile[0]
        x = [i/10**6 for i in x]
        # y = [i/10**6 for i in x]      
        y = datafile[1]

        tools.xy_plot([[x, y[index]]], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Frame {}'.format(str(index)), box = False, save = False, target_dir = self.target_dir)
        
    def plot_spectrum(self, xdata, ydata, vmin, vmax, crop=None):
        '''
        Function: Plots 3D spectrum
        Input:
            xdata = single array of frequencies
            ydata = list of arrays of intensities for each frame
            vmin, vmax = minimum/maximum values defining spectrum range
        Output:
            spectrum plot
        '''
        xdata = [d/10**6 for d in xdata]
        self.spectrum = method.spectrum(xdata, ydata, vmin=vmin, vmax=vmax, crop=crop, save=True, target_dir = self.target_dir)

        
    def correct_file(self, data_filename, background_filename):
        '''
        Function: Removes the average background from the data
        Input:
            data_filename = string of data file to correct
            background_filename = string of bacground file for correction
        Output:
            data_corr = background corrected data
        '''
        if background_filename is not None:
            data_corr = method.correct_background(self.data_dictionary, data_filename, background_filename)
            self.data_dictionary[str(data_filename)+'_corr'] = data_corr
        else:
            print('No background file available')
            data_corr = None

        return data_corr
    
        
    def process_data(self, datafile, crop, n_avg, width, height, distance, n_frames, plot=False):
        '''
        Function: Fits beatnotes in each frame with a multi-lorentzian function
        Input:
            datafile = data to analyse
            crop = select frequency domain within each frame to analyse
            n_avg = moving average, determines length of averaging window
            width = minimum peak width
            height = minimum peak height
            distance = minimum distance between neighbouring peaks
            n_frames = range of frames to analyse, defined as a list [index1, index2]
            plot = True/False, set to False to speed up function
        Output:
            amplitudes, frequencies, widths, fits = fit parameters
            saves fit parameters in binary .npy file
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

        amplitudes, frequencies, widths, fits = method.cycle_fit(xdata, ydata, n_avg, self.n_frames, minfit=False, threshold=None, width=width, height=height, distance=distance, prominence=None, target_dir = self.target_dir, plot=plot)

    def load_fit_parameters(self):
        '''
        Function: Loads all saved 'fit_parameters.npy' files into dictionary
        Output:
            fit_params = data dictionary with fit parameters from all saved files
        '''
        filenames = classify.mac_natsorted(os.listdir(self.target_dir))
        filenames = [f for f in filenames if 'fit_parameters' in f and '._' not in f]
        print('Fit parameters available to load: ', filenames)
        
        fit_params = {}
        for f in filenames:
            fit_params[f] = np.load(self.target_dir+f, allow_pickle=True)
        return fit_params
    
    def plot_timelines(self, fit_params, variable_type=None):
        '''
        Function: Plot beatnote parameters over all frames
        Input:
            fit_params = from saved file, (frames, amplitudes, frequencies, widths, fits)
            variable_type = optional, to select individual parameter to plot
        Output:
            timelines plot
        '''
        self.frames, self.amplitudes, self.frequencies, self.widths, self.fits = fit_params
        
        x = self.frames
        frame_no = len(self.frames)
        beat_no = len(self.frequencies)
        
        if variable_type == 'frequency':
            tools.xy_plot( [x, tools.convert_list(self.frequencies, 10**-6)], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Beat Frequency (MHz)', title = 'Beatnote Frequency over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)
        if variable_type == 'amplitude':
            tools.xy_plot( [x, self.amplitudes], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Amplitude (dB)', title = 'Beatnote Amplitude over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)
        if variable_type == 'width':
            tools.xy_plot( [x, tools.convert_list(self.widths, 10**-6)], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Width (MHz)', title = 'Beatnote Width over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)
        if variable_type == None:
            tools.xy_plot( [x, tools.convert_list(self.frequencies, 10**-6)], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Beat Frequency (MHz)', title = 'Beatnote Frequency over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)            
            tools.xy_plot( [x, self.amplitudes], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Amplitude (dB)', title = 'Beatnote Amplitude over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)
            tools.xy_plot( [x, tools.convert_list(self.widths, 10**-6)], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(beat_no)], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Width (MHz)', title = 'Beatnote Width over {} frames'.format(str(frame_no)), box = False, save = True, target_dir = self.target_dir)

        
    def filter_outliers(self, data, sigma = 3):
        '''
        Function: Filters any outliers in data above and below sigma number of standard deviationa
        Input:
            data = data to filter
            sigma = number of standard deviations to filter
        Output:
            data_filtered = filtered data by sigma
        '''
        self.data_filtered = method.filter_outliers(data, sigma)
        tools.xy_plot( self.data_filtered, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(data))], aspect = 0.5, yerror = None, x_label = 'Frame', y_label = 'Filtered Variable', title = 'Filtered Beatnote Variable', box = False, save = True, target_dir = self.target_dir)

        return self.data_filtered
        
        
    def get_beatnote_stats(self, x, beatnote_frequencies, beatnote_index=None, sigma=3, n_bins=30):
        '''
        Function: Selects frequency shifts above a certain number of standard deviations and plots them in a histogram
        Input:
            x = list of times
            frequencies = list of frequencies
            beatnote_index = None/i, which beatnote to analyse
            sigma = threshold number of standard deviations
            n_bins = number of histogram bins
        Output:
            selected_beats = filtered beatnote frequency shifts
            selected_times = corresponding times
        '''
        # tools.xy_plot(select_data, type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(self.frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = 'Beatnote Frequencies over {} frames'.format(str(len(self.t))), box = False, save = True, target_dir = self.target_dir)
        self.selected_times, self.selected_beats = method.select_higher_beatnotes(x, beatnote_frequencies, beatnote_index, sigma, self.target_dir)
        
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
