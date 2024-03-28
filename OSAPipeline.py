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

class OSAPipeline():
    
    def __init__(self, directory, folder, target_dir):
        self.directory = directory
        self.folder = folder
        self.target_dir = target_dir
        
    def load_data(self, mac = False):
        self.data_dictionary = osam.load_OSA_data(self.directory+self.folder, self.target_dir, mac = mac)

        return self.data_dictionary
        
    def plot_spectrum(self, filename):
        dataset = self.data_dictionary[str(filename)]
        
        x = dataset[0]
        y = dataset[1]
        #plt.plot(x, y)
        #plt.xlabel('Wavelength (nm)')
        #plt.ylabel('Intensity (a.u.)')
        title = 'OSA Plot {}'.format(str(filename))
        #plt.title(title)
        #plt.show()
        
        tools.xy_plot([dataset], fit = None, label_variable = None, aspect = 0.5, yerror = None, x_label = 'Wavelength (nm)', y_label = 'Intensity (a.u.)', title = title, box = False, save = False, target_dir = self.target_dir)

    
    def find_peak_spacing(self, osax, osay, filename, window_size, width=None, height=None, distance=None, save=False):
        
        osay = np.array(osay)
        osayc = osay - min(osay)
        osayc = osay

        plt.plot(osax, osayc)
        plt.show()

        lambdas = []
        intensities = []

        nwindow = int(len(osax)/window_size)
        for n in range(0, nwindow):
            xmin = n * window_size
            xmax = (n+1) * window_size

            maxima, minima = tools.peak_finder(osax[xmin:xmax], osayc[xmin:xmax], minfit=False,
                                               threshold=None, width=width, height=height, distance=distance, prominence=None, plot=True)

            lambdas.extend([osax[xmin:xmax][m] for m in maxima])
            intensities.extend([osayc[xmin:xmax][m]+min(osay) for m in maxima])
            # plt.scatter(lambdas, intensities)

        print(len(lambdas), ' peak maxima found at t = ', lambdas)
        for j, s in enumerate(lambdas):
            if j == 0:
                plt.plot((s, s), (min(osay), intensities[j]), 'tab:blue', zorder=2, label='lasing peaks')
            else:
                plt.plot((s, s), (min(osay), intensities[j]), 'tab:blue', zorder=2)
        plt.plot(osax, osay, linewidth=0.5, alpha=0.5,
                 color='lightgrey', label='spectrum')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.ylim(min(osay, 0))
        plt.legend()
        title = 'OSA Plot Sidebands {}'.format(str(filename))
        plt.title(title)
        plt.savefig(self.target_dir+str(title)+"_plot.png",
                    bbox_inches='tight', pad_inches=0.0, format='png', dpi=1200)
        plt.show()

        delta_lambdas = []
        for i in range(1, len(lambdas)):
            delta = lambdas[i] - lambdas[i-1]
            delta_lambdas.append(delta)

        print('Intensities: ', intensities)
        print('Delta Lambda: ', delta_lambdas)

        if save is True:
            np.save(self.target_dir+"{}_parameters.npy".format(filename),
                    np.array([lambdas, intensities, delta_lambdas], dtype=object))

        return lambdas, intensities, delta_lambdas