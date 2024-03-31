#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:50:47 2024

@author: sabrinaperrenoud
"""
#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np

import tools as tools
import classifier as classify
import ResonanceMethods as rem


'''
RESONANCE PIPELINE

Data analysis pipeline for ResonanceWorkbook
Parent class calling on ResonanceMethods functions
'''

class ResonancePipeline():
    
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
        
    def load_data(self, type='mat', mac = True):
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
            self.data_dictionary = classify.process_dict(self.data_folder)
        if type == 'csv':
            self.data_dictionary = classify.load_csv_file(self.directory+self.folder+'/', mac=mac)
        if type == 'txt':
            self.data_dictionary = classify.load_txt_file(self.directory+self.folder+'/', skiprows=14, mac=mac)
            
    def plot_data(self, x, y):
        '''
        Function: Simple x,y plot
        Input:
            x, y
        Output:
            single data plot
        '''
        tools.xy_plot([x, y], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = None, y_label = None, title = None, box = False, save = False, target_dir = self.target_dir)

    def classify_data(self, datafile):
        '''
        Function: Classify data in dictionary as 'time', 'resonance', and 'fwhm
        Input:
            datafile = specific dataset to calssify
        Output:
            data_parameters = dictionary with classified data
        '''
        self.data_parameters = {}
        
        self.data_parameters['time'] = datafile[:, 0]
        self.data_parameters['resonance'] = datafile[:, 1]
        self.data_parameters['fwhm'] = datafile[:, 2]
        
        tools.xy_plot([[self.data_parameters['time'], self.data_parameters['resonance']]], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Time (s)', y_label = r'\lambda (nm)', title = 'Wavelength Raw Data', box = False, save = True, target_dir = self.target_dir)
        tools.xy_plot([[self.data_parameters['time'], self.data_parameters['fwhm']]], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Time (s)', y_label = 'FWHM (nm)', title = 'FWHM Raw Data', box = False, save = True, target_dir = self.target_dir)

        return self.data_parameters
    
    def remove_errors(self, x, y_name, error_range):
        '''
        Function: Replaces data with nan in given error range
        Input:
            x = time
            y_name = specifies which data type in data_parameters dictionary to correct
            error_range = range of indices to remove, as list [inde1, index2]
        Output:
            y_corr = data with errors removed
        '''
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
        '''
        Function: Finds average moving background level and subtracts from data within given window
        Input:
            data_parameter = string corresponding to data in data_parameters dictionary
            window_size = over which to calculate average background, adjust for best results
        Output:
            y_corr = background corrected data
        '''
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

    
    def find_signal(self, x, y, type, interval, window_size, dev, distance, width):
        '''
        Function: Convolves data with step/spike kernel to identify step/spike locations and heights
        Input:
            x, y = data to analyse
            type = 'steps'/'spikes', which signal type to look for
            interval = domain of data to analyse
            window_size = size of domain over which to apply convolution
            dev = minimum number of standard deviations for picking up signal event heights
            distance = minimum separation between neighbouring signal events
            width = minimum signal duration
        Output:
            maxima, minima, signal_indices, signal_heights, signal_excluded, convolutions = result of convolution
        '''
        datax = x[interval[0] : interval[1]]
        datay = y[interval[0] : interval[1]]

        maxima, minima, signal_indices, signal_heights, signal_excluded, convolutions = rem.signal_finder(interval, datax, datay, window_size = window_size, type = type, minfit = True, distance = distance, width = width, height = None, dev = dev, plot = True, target_dir = self.target_dir)
        
    def load_signal_parameters(self, filename, type):
        '''
        Function: Loads signal convolution parameters in filename
        Input:
            filename = name of file to load
            type = 'steps'/'spikes', which signal type to look for
        Output:
            maxima, minima, signal_indices, signal_heights, signal_excluded, convolutions = result of convolution
        '''
        self.signal_parameters = {}
                       
        maxima, minima, signal_indices, signal_heights, signal_excluded, convolutions = np.load(self.target_dir+filename, allow_pickle=True)

        self.signal_parameters['maxima'] = maxima
        self.signal_parameters['minima'] = minima
        self.signal_parameters[str(type)+'_indices'] = signal_indices
        self.signal_parameters[str(type)] = signal_heights
        self.signal_parameters[str(type)+'_excluded'] = signal_excluded
        self.signal_parameters['convolutions'] = convolutions
                     
    
    def dev_selection(self, i, y, stype = 'both', dev = 3, plot = True):
        '''
        Function: Filters data above and below sigma standard deviations
        Input:
            i, y = data to filter
            stype = 'both'/'positive'/'negative'
            sigma = threshold number of standard deviations
        Output:
            data_filtered = filtered data
        '''    
        std = np.nanstd(y)
        mu = np.nanmean(y)
        h_lim = float(std) * dev
        
        iselected = []
        selected = []
        for m, n in zip(i, y):
            if stype == 'both':
                if n >= h_lim or n <= - h_lim:
                    selected.append(n)
                    iselected.append(m)
            elif stype == 'positive':
                if n >= h_lim:
                    selected.append(n)
                    iselected.append(m)
            elif stype == 'negative':
                if n <= - h_lim:
                    selected.append(n)
                    iselected.append(m)
                    
        if plot is True:
            tools.xy_plot([[i, y], [iselected, selected]], type='beat_timelines', label_variable = ['total steps', r'filtered steps {}$\sigma$'.format(dev)], aspect = 0.5, yerror = None, x_label = 'Index', y_label = r'$\Delta \lambda$ (nm)', title = 'Data Filter {} sigma'.format(dev), box = False, save = True, target_dir = self.target_dir)
        return iselected, selected

    def fit_histogram(self, hist_data, plot=False):
        '''
        Function: Fits histogram to data
        Input:
            his_data = data to fit
            stype = 'both'/'positive'/'negative'
            sigma = threshold number of standard deviations
        Output:
            data_filtered = filtered data
        ''' 
        hist_data_nan = [h for h in hist_data if str(h) != 'nan']
        hist = hist_data_nan

        x = np.linspace(min(hist), max(hist), 1000)

        mu, std = norm.fit(hist)
        p = norm.pdf(x, mu, std)
        h_bins = 100
        a = max(p)

        
        if plot is True:
            tools.xy_plot([hist, h_bins], type = 'histogram', fit=[x, p], label_variable = r'$\mu=$ {}, $\sigma=$ {}'.format(round(mu,7), round(std,8)), aspect = 0.8, yerror = None, x_label = r'Step $\Delta \lambda$ (nm)', y_label = 'n', title = '1D Histogram: Normal Fit', box = False, save = True, target_dir = self.target_dir)
        return mu, std, a

    def fit_gaussian(self, hist_data, ind, save = False):
        '''
        Function: Fits gaussian function to data
        Input:
            his_data = data to fit
        Output:
            fit_params, single_gaussian = gaussian fit output
        ''' 
        fit_params, single_gaussian = rem.single_gaussian_fit(hist_data, ind = 0, save = True, target_dir = self.target_dir)
        
        return fit_params, single_gaussian
    
        
    def get_stats(self, x, y, n_bins = None, split = True, abso = True):
        '''
        Function: Plot histograms
        Input:
            x, y = data to plot
            n_bins = number of histogram bins
            split = plot negative and positive shifts
            abso = plot combined absolute values for positive and negative shifts
        Output:
            histogram plots
        ''' 
        
        self.hist_data = y
        
        if n_bins is None:
            n_bins = len(self.hist_data)
        tools.xy_plot([self.hist_data,n_bins], type = 'histogram', aspect = 1.0, yerror = None, x_label = r'Step $\Delta \lambda$ (nm)', y_label = 'n', title = '1D Histogram', box = False, save = True, target_dir = self.target_dir)
        
        if split is True:
            self.hist_data_pos = [h for h in self.hist_data if h>0]
            self.hist_data_neg = [h for h in self.hist_data if h<0]
            
            n_bins1 = len(self.hist_data_pos)
            tools.xy_plot([self.hist_data_pos,n_bins1], type = 'histogram', aspect = 1.0, yerror = None, x_label = r'Step $\Delta \lambda$ (nm)', y_label = 'n', title = '1D Histogram: Positive Values', box = False, save = True, target_dir = self.target_dir)
             
            n_bins2 = len(self.hist_data_neg)
            tools.xy_plot([self.hist_data_neg,n_bins2], type = 'histogram', aspect = 1.0, yerror = None, x_label = r'Step $\Delta \lambda$ (nm)', y_label = 'n', title = '1D Histogram: Negative Values', box = False, save = True, target_dir = self.target_dir)
        
        if abso is True:
            self.hist_data_abs = [np.abs(h) for h in self.hist_data]
            tools.xy_plot([self.hist_data_abs, n_bins], type = 'histogram', aspect = 1.0, yerror = None, x_label = r'Step $\Delta \lambda$ (nm)', y_label = 'n', title = '1D Histogram: Absolute Values', box = False, save = True, target_dir = self.target_dir)
   
    def get_temporal_stats(self, y, step_indices, steps, window, h_bins = None, fit_poisson = False):
        '''
        Function: Counts number of detected signal events in given window, plots in histogram and fits poisson distribution
        Input:
            y = wavelengths
            step_indices = locations of steps/spikes
            steps = step/spike heights
            window = over which to count number of signal events
            h_bins = number of histogram bins
            fit_poisson = True/False, fits poisson distribution to histogram counts data
        Output:
            counts = number of signal events in each window
            poisson_fit = fitted poisson distribution
        ''' 
        window_no = int(len(y)/window)
        counts = rem.get_counts(step_indices, steps, window, window_no)

        if h_bins is None:
            h_bins = len(counts)
        tools.xy_plot([counts, h_bins], type = 'histogram', fit=None, label_variable = None, aspect = 0.8, yerror = None, x_label = 'Detection Counts', y_label = 'n', title = 'Counts Histogram over time', box = False, save = True, target_dir = self.target_dir)
        
        print('Counts list across {} windows of size {}:'.format(window_no, window))

        if fit_poisson is True:
            poisson_fit = rem.poisson_fit_histogram(counts)

        return counts, poisson_fit
