#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os

import numpy as np
from matplotlib.ticker import (MultipleLocator,AutoMinorLocator, FormatStrFormatter, ScalarFormatter)
import matplotlib.cm as cm
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
import numba
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
import scipy.odr as odr
from scipy import stats
from scipy import signal
from scipy.signal import correlate
from itertools import zip_longest

import tools as tools
import fitting as fit



def load_OSA_data(directory, target_dir, mac = False):
    '''
    Function: Loads all .csv files in target folder
    Input:
        directory = to fodler containing .csv OSA data files
        target_dir = output folder
        mac = True/False
    Output:
        data_dictionary = dictionary of data for all OSA .csv files
    '''
    if mac is True:
        datasetnames = classify.mac_natsorted(os.listdir(directory))
    else:
        datasetnames = os.listdir(directory)

    datasetnames = [d for d in datasetnames if '._' not in d]
    print('Available datasets: ', datasetnames)

    data_dictionary = {}
    for d in datasetnames:
        df = pd.read_csv(directory+d, sep='\t')
        '''
        OSA spectrum under column name: '73CSV  ', line 31+
        '''
        #print(list(df.columns.values))
        df = df['73CSV  ']

        datadf = df.iloc[0:][31:]
        datadf = np.array([f for f in datadf])
        x = []
        y = []
        for a in datadf:
            b, c = a.split(',')
            x.append(float(b))
            y.append(float(c))
        #datasets.append([x, y])
        
        dname = d.strip('.CSV')
        data_dictionary[dname] = [x, y]
        
        plt.plot(x, y)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        title = 'OSA Plot {}'.format(str(d.strip('.CSV')))
        plt.title(title)
        plt.savefig(target_dir+str(d.strip('.CSV'))+"_plot.png",
                    bbox_inches='tight', pad_inches=0.0, format='png', dpi=1200)
        plt.show()
    return data_dictionary


def find_peak_spacings(osax, osay, window_size, width, height, distance, filename, save, target_dir):
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
            plt.plot((s, s), (min(osay), intensities[j]), 'tab:blue', zorder=1, label='lasing peaks')
        else:
            plt.plot((s, s), (min(osay), intensities[j]), 'tab:blue', zorder=1)
    plt.plot(osax, osay, linewidth=0.5, alpha=0.5, color='lightgrey', label='spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.ylim(min(osay), 0)
    plt.legend()
    title = 'OSA Plot {}'.format(str(filename))
    plt.title(title)
    plt.savefig(target_dir+str(title)+"_plot.png",
                bbox_inches='tight', pad_inches=0.0, format='png', dpi=1200)
    plt.show()
    plt.show()

    delta_lambdas = []
    for i in range(1, len(lambdas)):
        delta = lambdas[i] - lambdas[i-1]
        delta_lambdas.append(delta)

    print('Intensities: ', intensities)
    print('Delta Lambda: ', delta_lambdas)

    if save is True:
        np.save(target_dir+"{}_parameters.npy".format(filename),
                np.array([lambdas, intensities, delta_lambdas], dtype=object))

    return lambdas, intensities, delta_lambdas

def odr_auto_lorentz(xdata, ydata, x0, gamma_guess=0.01):
    '''
    Function: ODR  for Lorentzian fit
    Input:
        xdata, ydata =  data to fit
        x0 = peak position to fix
        gamma_guess = fit guess for lorentzian width
    Output:
        output = optimised Lorentzian fit parameters
    '''   
    print(x0)
    print(len(xdata))
    print(x0)
    offset_guess = min(ydata)
    a_guess = max(ydata)
    x0_guess = x0
    pguess = [a_guess, x0_guess, gamma_guess, offset_guess]

    
    param_mask = np.ones(len(pguess))
    param_mask[1] = 0
    # param_mask[2] = 0
    
    popt, pcov = tools.odrfit(fit.odr_lorentzian_lineshape, xdata, ydata, initials = pguess, param_mask = param_mask)
    print('Guess = ',pguess)
    print('Optimsed Parameters = ',popt)
    
    output = popt

    return output


def fit_lasing_modes(osax, osay, indlocs, gamma_guess, save = False, target_dir=None):
    '''
    Function: Cycles through all peak locations to fit with lorentzian lineshape
    Input:
        osax, osay =  data to fit
        indlocs = index locations of identified peaks
        gamma_guess = fit guess for lorentzian width
        save = True/False
    Output:
        lorentz_coeffs = optimised Lorentzian fit parameters
        lorentz_fits = lorentzian fits
    '''
    local = fit.get_local_indices(indlocs, len(osay))
    print(local)

    lorentz_coeffs = []
    lorentz_fits = []
    linewidths = []
    for i, loc in enumerate(indlocs):
        print(min(local[i]), max(local[i]))
        print(i, loc)
        plt.plot(osax[int(min(local[i])):int(max(local[i]))], osay[int(min(local[i])):int(max(local[i]))])
        lorentz_coeff = odr_auto_lorentz(osax[int(min(local[i])):int(max(local[i]))], osay[int(min(local[i])):int(max(local[i]))], int(loc-min(local[i])))
        lorentz_coeffs.append(lorentz_coeff)
        lorentz_fit = fit.odr_lorentzian_lineshape(lorentz_coeff, osax[int(min(local[i])):int(max(local[i]))])
        lorentz_fits.append(lorentz_fit)

        print('Lorentz fit coefficients (A, mu, gamma, offset = ', lorentz_coeff)
        linewidth = tools.get_linewidth(lorentz_coeff[1], lorentz_coeff[2])
        linewidths.append(linewidth)
        plt.plot(osax[int(min(local[i])):int(max(local[i]))], lorentz_fit, label=r'$\delta\omega_{}=$'.format(i+1)+str(round(linewidth*10**-9,2))+' GHz', linewidth=0.8, zorder=3, color='tab:blue')
        plt.show()
    plt.plot(osax, osay, label = 'OSA signal', color = 'lightgrey', linewidth = 0.5)

    plt.ylim(min(osay), max(osay))
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    title = 'OSA Spectrum Lasing Modes Lorentz Fit'
    plt.title(title, fontsize = 8)
    if save is True:
        plt.savefig(target_dir+str(title)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=1200)
    plt.show()

    return lorentz_coeffs, lorentz_fits, linewidths
