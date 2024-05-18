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

import OSAMethods as osam

class OSAPipeline():
    
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
        
    def load_data(self, mac = False):
        '''
        Function: Loads all data in folder
        Input:
            mac = True/False
        Output:
            data_dictionary = stores all data under associated filenames
        '''
        self.data_dictionary = osam.load_OSA_data(self.directory+self.folder, self.target_dir, mac = mac)

        return self.data_dictionary
        
    def plot_spectrum(self, filename):
        '''
        Function: Plots OSA spectrum
        Input:
            filename = specific dataset in data_dictionary to plot
        Output:
            spectrum plot
        '''
        dataset = self.data_dictionary[str(filename)]
        
        x = dataset[0]
        y = dataset[1]
        title = 'OSA Plot {}'.format(str(filename))

        tools.xy_plot([dataset], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Wavelength (nm)', y_label = 'Intensity (a.u.)', title = title, box = False, save = False, target_dir = self.target_dir)

    
    def find_peak_spacing(self, osax, osay, filename, window_size, width=None, height=None, distance=None, save=False):
        '''
        Function: Find position and spacing of lasing modes in spectrum 
        Input:
            osax = x values of spectrum (wavelength in nm)
            osay = y values of spectrum (intensity)
            window_size = defines domain over which to search for peaks
            width, height, distance = peak finding parameters
            filename = name of data file, used for saving data under relevant title
            save = True/False
            target_dir = output folder
        Output:
            lambdas = wavelengths of identified peaks
            intensities = heights of identified peaks
            delta_lambdas = wavelength spacing between neighbouring peaks
        '''
        osayc = []
        for yvalues in osay:
            if yvalues < -100:
                osayc.append(-100)
            else:
                osayc.append(yvalues)


        lambdas, intensities, delta_lambdas = osam.find_peak_spacings(osax, osay, window_size, width, height, distance, filename, save, self.target_dir)

        return lambdas, intensities, delta_lambdas

    def fit_lasing_mode(self, osax, osay, peak_locations, gamma_guess, save = False):
        '''
        Function: Cycles through all peak locations to fit with lorentzian lineshape
        Input:
            osax, osay =  data to fit
            peak_locations = locations of identified peaks
            gamma_guess = fit guess for lorentzian width
            save = True/False
        Output:
            lorentz_coeffs = optimised Lorentzian fit parameters
            lorentz_fits = lorentzian fits
        '''
        indlocs = []
        for i,j in enumerate(osax):
            if j in peak_locations:
                indlocs.append(i)
        print(indlocs)
        
        lorentz_coeffs, lorentz_fits, linewidths = osam.fit_lasing_modes(osax, osay, indlocs, gamma_guess, save = False, target_dir = self.target_dir)
        
        return lorentz_coeffs, lorentz_fits, linewidths
