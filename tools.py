#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:55:22 2024

@author: sabrinaperrenoud
"""
import matplotlib.pyplot as plt
import os
from natsort import natsorted
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
from scipy.stats import norm





def xy_plot(datasets, type=None, fit = None, label_variable = None, aspect = 1.0, yerror = None, x_label = 'X Axis', y_label = 'Y Axis', title = None, box = False, save = False, target_dir = 'output/'):
    if type == '3d':
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (9,9 * aspect))
        ax.set_box_aspect((1*aspect,1 ,1 ))
        ax.view_init(40, 40)
        fig.set_facecolor('white')
        ax.set_facecolor('white') 

        ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
        ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
        ax.tick_params('z', labelsize=14, pad = 15)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.zaxis.set_minor_locator(AutoMinorLocator(2))

        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.zaxis.set_major_formatter(ScalarFormatter())

        ax.set_xlabel(x_label,fontsize=18,labelpad = 15)
        ax.set_ylabel(y_label,fontsize=18,labelpad = 15)
        # ax.set_zlabel(r'$g^{(2)}(\tau)$)', fontsize=18,labelpad = 25)
        
        for j in range(len(datasets[0])):
            y = np.ones(len(datasets[0][0]))*j
            z = datasets[1][j]
            x = datasets[0][0]
         
            ax.plot3D(x, y, z)
            
    elif type == 'histogram':
        fig, ax = plt.subplots(figsize = (9,9 * aspect))
    
    
        ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
        ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
        ax.set_xlabel(x_label,fontsize=18,labelpad = 15)
        ax.set_ylabel(y_label,fontsize=18,labelpad = 15)
    
        # if len(fit) >0:
        #     plt.hist(datasets[0], density = True, bins = datasets[1], label = 'Histogram')
        #     plt.plot(fit[0], fit[1], 'k', linewidth=2, linestyle='--', label=label_variable)
        if fit==None:
            plt.hist(datasets[0], density = True, bins = datasets[1], label = label_variable)
        else:
            plt.hist(datasets[0], density = True, bins = datasets[1], label = 'Histogram')
            plt.plot(fit[0], fit[1], 'k', linewidth=2, linestyle='--', label=label_variable)
        
        # xabs_max = abs(max(ax.get_xlim(), key=abs))
        # ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
        # ax.set_xlim(xmin=-2, xmax=2)
        
        # ax.axvline(x = 0, color = 'black', linewidth = 0.8, alpha = 0.5)

    elif type == 'multi_histogram':
        fig, ax = plt.subplots(figsize = (9,9 * aspect))
    
    
        ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
        ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
        ax.set_xlabel(x_label,fontsize=18,labelpad = 15)
        ax.set_ylabel(y_label,fontsize=18,labelpad = 15)
        
        for i, data in enumerate(datasets):
            if fit==None:
                plt.hist(data[0], density = True, bins = data[1], label = label_variable[i])
            else:
                plt.hist(data[0], density = True, bins = data[1], label = 'Histogram')
                plt.plot(fit[0], fit[1], 'k', linewidth=2, linestyle='--', label=label_variable[i])
        
        xabs_max = abs(max(ax.get_xlim(), key=abs))
        # ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
        # ax.set_xlim(xmin=-2, xmax=2)
        
        ax.axvline(x = 0, color = 'black', linewidth = 0.8, alpha = 0.5)
        
        
    elif type == 'timeline':
        fig, ax = plt.subplots(figsize = (9,9 * aspect))
    
    
        ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
        ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
        ax.set_xlabel(x_label,fontsize=18,labelpad = 15)
        ax.set_ylabel(y_label,fontsize=18,labelpad = 15)
        
        
        for i, data in enumerate(datasets):
            if fit is None and label_variable is None:
                plt.plot(data[0], data[1], label = None)
            elif fit is None:
                plt.plot(data[0], data[1], label = label_variable[i])
            else:
                plt.plot(data[0], data[1], color = 'black', linestyle = '--', label = 'Data')
                plt.plot(data[0], fit[i], color = 'red', label = 'Fit')
        # plt.ylim(0,0.2)
    elif type == 'beat_timeline':
        fig, ax = plt.subplots(figsize = (9,9 * aspect))
    
    
        ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
        ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
        ax.set_xlabel(x_label,fontsize=18,labelpad = 15)
        ax.set_ylabel(y_label,fontsize=18,labelpad = 15)
    
        
        for i, data in enumerate(datasets[1]):
            if fit is None and label_variable is None:
                if len(datasets[0]) == 1:
                    plt.scatter(datasets[0], data, label = None, marker='.')
                else:
                    plt.scatter(datasets[0][i], data, label = None, marker='.')
                # plt.ylim(0,25)
            elif fit is None:
                plt.scatter(datasets[0], data, label = label_variable[i], marker='.')
                # plt.ylim(0,35)
        
            else:
                plt.scatter(datasets[0], data, color = 'grey', label = 'Data')
                xfit = np.linspace(min(datasets[0]), max(datasets[0]), len(fit[i]))
                plt.plot(xfit, fit[i], color = 'tab:blue', label = 'Fit')

    elif type == 'beat_timelines':
        fig, ax = plt.subplots(figsize = (9,9 * aspect))
    
    
        ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
        ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
        ax.set_xlabel(x_label,fontsize=18,labelpad = 15)
        ax.set_ylabel(y_label,fontsize=18,labelpad = 15)
    
        
        for i, data in enumerate(datasets):
            if fit is None and label_variable is None:
                plt.scatter(data[0], data[1], label = None, marker='.')
                # plt.ylim(0,25)
            elif fit is None:
                plt.scatter(data[0], data[1], label = label_variable[i], marker='.')
                # plt.ylim(0,35)
        
            else:
                plt.scatter(data[0], data[1], color = 'grey', label = 'Data')
                xfit = np.linspace(min(datasets[0]), max(datasets[0]), len(fit[i]))
                plt.plot(xfit, fit[i], color = 'tab:blue', label = 'Fit')
    
    elif type == 'model_efficiency':
        fig, ax = plt.subplots(figsize = (9,9 * aspect))
    
    
        ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
        ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
        ax.set_xlabel(x_label,fontsize=18,labelpad = 15)
        ax.set_ylabel(y_label,fontsize=18,labelpad = 15)
        
        # plt.ylim(0,1)
        
        for i, data in enumerate(datasets):
            if fit is None:
                plt.plot(data[0], data[1], color = 'black')
            else:
                plt.plot(data[0], data[1], color = 'grey', label = 'Estimated mean')
                plt.plot([0, 1], [fit[0], fit[0]], color = 'red', label = 'True mean')
                plt.plot([fit[1], fit[1]], [0, fit[0]], color = 'blue', label = 'Theoretical efficiency')
        
        
    else:
    
    
        fig, ax = plt.subplots(figsize = (9,9 * aspect))
    
    
        ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
        ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
        ax.set_xlabel(x_label,fontsize=18,labelpad = 15)
        ax.set_ylabel(y_label,fontsize=18,labelpad = 15)
        
        # plt.ylim(0,30)
        
        for i, data in enumerate(datasets):
            if fit is None:
                if label_variable is None:
                    plt.plot(data[0], data[1])
                else:
                    plt.plot(data[0], data[1], label = label_variable[i])
            else:
                plt.plot(data[0], data[1], color = 'grey', label = 'Data')
                xfit = np.linspace(min(data[0]), max(data[0]), len(fit[i]))
                plt.plot(xfit, fit[i], color = 'tab:blue', label = 'Fit')

    plt.legend()
    plt.title(title)
    
    if save:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        plt.savefig(target_dir+str(title.replace(" ", "_"))+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi = 1200)
        # plt.savefig(target_dir+str(title.replace(" ", "_"))+"_plot.eps", bbox_inches='tight',pad_inches=0.0, format = 'eps')
        plt.show()
    else:
        plt.show()

def background_avg(bg):
    frequency = bg[0]
    frequency = [f/10**6 for f in frequency]
    signal = bg[1]
    
    sum_s = np.sum(signal,0)
    
    avg_s = []
    
    for s in sum_s:
        average = s/len(signal)
        avg_s.append(average)

    datasets = [[frequency, np.array(avg_s)]]
    xy_plot(datasets, aspect = 0.8, title = 'Background', x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)')


    return avg_s

def frequency_crop(val, frequency):
    frequency_spacing = (max(frequency)-min(frequency))/len(frequency)
    frequency_crop = int(val / frequency_spacing)
    return frequency_crop


        
def moving_average(arr, window_size, factor=None, floor = False):
     
    i = 0
    moving_averages = []
    

    while i < len(arr) - window_size + 1:

        window = arr[i : i + window_size]
        window_average = round(sum(window) / window_size, 2)
        
        floor_window_size = window_size*factor
        floor_window = arr[i : i + floor_window_size]
        
        moving_floor = round(sum(floor_window) / floor_window_size, 2)
        
        if floor is True:

            moving_averages.append(window_average - moving_floor)
        
        else:
            moving_averages.append(window_average)

        i += 1
     
    # print(moving_averages)
    return moving_averages

def moving_average2(arr, window_size, factor=None, floor = False):
     
    i = 0
    moving_averages = []
    

    while i < len(arr) - window_size + 1:

        window = arr[i : i + window_size]
        window_average = sum(window) / window_size
        
        floor_window_size = window_size*factor
        floor_window = arr[i : i + floor_window_size]
        
        moving_floor = round(sum(floor_window) / floor_window_size, 2)
        
        if floor is True:

            moving_averages.append(window_average - moving_floor)
        
        else:
            moving_averages.append(window_average)

        i += 1
     
    # print(moving_averages)
    return moving_averages
    


def average_y(coordinate_list, fstart, fend):
    y_values_by_x = {}
    for x, y in coordinate_list:
        y_values_by_x.setdefault(x, []).append(y)
    average_y_by_x = [sum(v)/len(v) for k, v in y_values_by_x.items()]
    
    xy_plot([[np.arange(0,len(average_y_by_x)), average_y_by_x]], aspect = 0.5, yerror = None, x_label = 'Frames', y_label = 'Frequency (MHz)', title = 'Averaged y-values', box = False, save = False, target_dir = 'output/')
    # plt.plot(average_y_by_x)
    # plt.ylim(fstart,fend)BSB2
    plt.show()
    return average_y_by_x

def peak_finder(datax, data, minfit = False, threshold=None, width=None, height=None, distance=None, prominence=None, plot=True):
    maxima = signal.find_peaks(data, height = height, threshold = threshold, width = width, distance=distance, prominence=None)[0]
    data2 = [-d for d in data]
    minima = signal.find_peaks(data2, height = height, threshold = threshold, width = width, distance=distance, prominence=prominence)[0]
    
    x_peak = [datax[m] for m in maxima]
    x_peak2 = [datax[m] for m in minima]
    
    print(len(maxima),' peak maxima found at t = ',x_peak)
    
    
    if minfit is True:
        
        print(len(minima),' peak minima found at t = ',x_peak2)
        for peak2 in minima:
            plt.axvline(x = datax[peak2], color = 'r')
        # plt.plot(-1*data2 + np.max(data2))
        # maxima = np.concatenate((maxima, minima), axis = 0)
        
    if plot is True:
        for peak in maxima:
            plt.axvline(x = datax[peak], color = 'b')
        # plt.plot(-1*data + np.max(data))
        
        datax = datax[:len(data)]
        plt.plot(datax, data, color = 'black', label = 'Signal Trace')
        plt.legend()
        plt.show()
    
    
    return maxima, minima




def fwhm(x):
    mx = max(x)+min(x)
    mxs = [i for i in range(len(x)) if x[i] > mx/2]
    return max(mxs) - min(mxs)

def get_diff(y):
    dx = 1
    ydiff = np.diff(y)
    dy = ydiff/dx
    return dy

def cross_correlation(a, b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    print(a)
    
    c = np.correlate(np.ma.array(a, mask=np.isnan(a)), np.ma.array(b,mask = np.isnan(b)),'full') 
    return c


def convert_list(datalist, factor):
    if len(datalist) == 1:
        datalist_convert = list(np.array(datalist)*factor)
    elif len(datalist) > 1:
        datalist_convert = []
        for dlist in datalist:
            dlist_convert = list(np.array(dlist)*factor)
            datalist_convert.append(dlist_convert)
    return datalist_convert


def wtof(w_nm, ghz=False):
    c = 299792458
    f = c/(w_nm*10**-9)
    if ghz is True:
        f = f*10**-9
    return f

def get_linewidth(l0, gamma):
    c = 299792458
    l1 = (l0 - gamma/2 )*10**-9
    l2 = (l0 + gamma/2 )*10**-9
    f1 = c/l1
    f2 = c/l2
    linewidth = np.abs(f1-f2)
    print(linewidth)
    return linewidth
