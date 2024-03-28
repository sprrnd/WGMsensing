#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:55:22 2024

@author: sabrinaperrenoud
"""
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

import tools as tools
from itertools import zip_longest

        
def spectrum(xdata, ydata, vmin=0, vmax=10, crop=None, save=False, target_dir = 'output/'):
    
    fig, ax = plt.subplots(figsize = (15,9))

    ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
    ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # xdata = [d/10**6 for d in xdata]
    
    if crop is None:
        data = np.transpose(ydata)
    else:
        ydata = ydata[crop[0] : crop[1]]
        data = np.transpose(ydata)
        

    x = np.linspace(crop[0], crop[1], len(ydata))
    y = np.linspace(min(xdata), max(xdata), len(xdata))

    
    plt.pcolormesh(x, y, data, shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')

    plt.colorbar()

    ax.set_xlabel('Frames',fontsize=18,labelpad = 15)
    ax.set_ylabel('Frequency (MHz)',fontsize=18,labelpad = 15)
    
    if save:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        plt.savefig(target_dir+"spectrum_plot_{}.png".format(str(crop)), bbox_inches='tight',pad_inches=0.0)
        plt.savefig(target_dir+"spectrum_plot_{}.eps".format(str(crop)), bbox_inches='tight',pad_inches=0.0)
        plt.show()
    else:
        plt.show()

def odr_lorentzian_lineshape(params, x):
    a, x0, gam, offset = params
    return a * gam**2 / ( gam**2 + ( x - x0 )**2) + offset

def odr_auto_lorentz(xdata, ydata, x0):
    
    print(x0)
    print(len(xdata))
    print(xdata[x0])
    offset_guess = min(ydata)
    a_guess = max(ydata)
    x0_guess = xdata[x0]
    gamma_guess = np.std(ydata)
    gamma_guess = 0.05 *10**6
    pguess = [a_guess, x0_guess, gamma_guess, offset_guess]

    
    param_mask = np.ones(len(pguess))
    param_mask[1] = 0
    # param_mask[2] = 0
    
    popt, pcov = tools.odrfit(odr_lorentzian_lineshape, xdata, ydata, initials = pguess, param_mask = param_mask)
    print('Guess = ',pguess)
    print('Optimsed Parameters = ',popt)
    
    output = popt

    return output

def odr_lorentz_func(params, x):
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        a = params[i]
        x0 = params[i+1]
        gam = params[i+2]
        offset = params[i+3]
        y = y + a * gam**2 / ( gam**2 + ( x - x0 )**2) + offset
    return y

def multi_lorentz_fit(x, data, maxima, minima, minfit, plot):

    local = get_local_indices(maxima, len(data))

    amplitudes = [[] for _ in range(len(maxima))]
    frequencies = [[] for _ in range(len(maxima))]
    widths = [[] for _ in range(len(maxima))]
    
    
    guess = ()
    print(guess)

    for i in range(len(maxima)):
        print('Fitting ',i+1,'out of ', len(maxima),'peaks')
        # print(local[i])
        print('Domain to fit: ',min(local[i]),max(local[i]))
        
        output = odr_auto_lorentz(x[min(local[i]):max(local[i])], data[min(local[i]):max(local[i])], maxima[i]-min(local[i]))
        print('OUTPUT',output)
        a, x0, gam, off = output

        
        frequencies[i].append(x[maxima[i]])
        amplitudes[i].append(a)
        widths[i].append(gam)
        guess += a, x0, gam, off

    
    if minfit is True:
        local2 = get_local_indices(minima, len(data))
        
        for j in range(len(minima)):
            print('Fitting ',j+1,'out of ', len(minima),'peaks')
            # print(local[i])
            print('Domain to fit: ',min(local2[j]),max(local2[j]))
            a, x0, gam, off = odr_auto_lorentz(x[min(local2[j]):max(local2[j])], data[min(local2[j]):max(local2[j])], minima[i]-min(local2[j]))
            # print(a, x0, gam, off)
            amplitudes.append(-a)
            widths.append(gam)
            guess += -a, x0+min(local2[j]), gam, off
        
        
    print('Guess: ',guess)
    
    param_mask = np.ones(len(guess))
    for k in range(int(len(guess)/4)-1):
        param_mask[1 + 4*k] = 0
    
    popt, pcov = tools.odrfit(odr_func, x, data, initials = guess, param_mask = param_mask)
    fit = odr_lorentz_func(popt, x)

    # popt, pcov = curve_fit(func, x, data, p0=guess)
    # fit = func(x, *popt)
    if plot is True:
        x = [x/10**6 for x in x]
        tools.xy_plot([[x, data]], fit = [fit], label_variable = None, aspect = 0.5, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Frame-by-frame Multi-Lorentzian Fit', box = False, save = True, target_dir = 'output/')

    print('Fit parameters [a, x0, gam, off]: ',popt)
    
    
    return amplitudes, frequencies, widths, fit

def cycle_fit(x, y, n_avg, n_frames, minfit, threshold, width, height, distance, prominence, target_dir, plot=False):
    fits = []
    frequencies = []
    amplitudes = []
    widths = []
    
    nn_frames = int(n_frames[1]) - int(n_frames[0])
    
    maximals = []
    
    
    for i, frame in enumerate(y[ int(n_frames[0]) : int(n_frames[1])] ):
        print('Fitting frame number ',i)
        
        # tools.xy_plot([[x, frame]], aspect = 0.5, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Raw data {}'.format(i), box = False, save = False, target_dir = 'output/')
        
        moving_averages = tools.moving_average(frame, n_avg, factor = 40, floor=False)
        x_ma = tools.moving_average(x, n_avg, factor = 40, floor=False)
        # tools.xy_plot([[x_ma, moving_averages]], aspect = 0.5, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Moving average {} for data {}'.format(n_avg,i), box = False, save = False, target_dir = 'output/')
        
        data = moving_averages
    
        maxima, minima = tools.peak_finder(x_ma, data, minfit=minfit, threshold=threshold, width=width, height=height, distance=distance, prominence = prominence, plot=plot)
        maximals.append(int(len(maxima)))
        
        if i == 0:
            amplitudes += [[] for _ in range(len(maxima))]
            frequencies += [[] for _ in range(len(maxima))]
            widths += [[] for _ in range(len(maxima))]


        if len(maxima) == 0:
            nanlist = [[np.nan]]*maximals[0]
            amplitude, frequency, wid, fit = nanlist, nanlist, nanlist, nanlist
            maxima = [np.nan] * len(amplitudes)
            fits.append(fit)

            
        else:
            amplitude, frequency, wid, fit = multi_lorentz_fit(x_ma, data, maxima, minima, minfit, plot=plot)
            fits.append(fit)
            
        
        
        for m in range(len(maxima)):
            if m+1 > len(amplitudes):
                print(i)
                amplitudes.append([np.nan] * (i - len(amplitudes[m-1])))
                frequencies.append([np.nan] * (i - len(frequencies[m-1])))
                widths.append([np.nan] * (i - len(widths[m-1])))
                print(len(amplitudes[m]))
                amplitudes[m] += amplitude[m]
                frequencies[m] += frequency[m]
                widths[m] += wid[m]
            else:
                amplitudes[m] += amplitude[m]
                frequencies[m] += frequency[m]
                widths[m] += wid[m]

    frames = np.arange(n_frames[0], n_frames[1]).tolist()
    amplitudes = [list(tpl) for tpl in zip(*zip_longest(*amplitudes, fillvalue = np.nan))]
    frequencies = [list(tpl) for tpl in zip(*zip_longest(*frequencies, fillvalue = np.nan))]
    widths = [list(tpl) for tpl in zip(*zip_longest(*widths, fillvalue = np.nan))]
    fits = [list(tpl) for tpl in zip(*zip_longest(*fits, fillvalue = np.nan))]
    
    np.save(target_dir+str(n_frames)+"_fit_parameters.npy", np.array([frames, amplitudes, frequencies, widths, fits], dtype = object))
    
    return amplitudes, frequencies, widths, fits  

def get_local_indices(peak_indices, domain):
    chunks = []
    
    for i in range(len(peak_indices)):
        
        peak_index = peak_indices[i]
        if i>0 and i<len(peak_indices)-1:
            distance = min(abs(peak_index-peak_indices[i+1]),abs(peak_indices[i-1]-peak_index))
        elif i==0:
            distance = abs(peak_index-peak_indices[i+1])
        else:
            distance = abs(peak_indices[i-1]-peak_index)
    
        if peak_index-int(distance/2) <=0:
            min_index = peak_index-int(distance/4)
            max_index = peak_index+int(distance/2)
        else:
            min_index = peak_index-int(distance/4)
            max_index = peak_index+int(distance/4)
        
        if min_index >= 0 and max_index <= domain:
            indices = np.arange(min_index,max_index)
            chunks.append(indices)
        elif min_index < 0 and max_index <= domain:
            min_dif = abs(min_index)
            indices = np.arange(min_index+min_dif,max_index)
            chunks.append(indices)
        elif min_index>=0 and max_index > domain:
            max_dif = abs(domain-max_index)
            indices = np.arange(min_index,max_index-max_dif)
            chunks.append(indices)            
        else:
            min_dif = abs(min_index)
            max_dif = abs(domain-max_index)
            indices = np.arange(min_index+min_dif,max_index-max_dif)
            chunks.append(indices)    
    
    return chunks

def filter_outliers(data, sigma):
    '''
    Function: Filters any outliers in data above and below sigma number of standard deviationa
    Input:
        data = data to filter
        sigma = number of standard deviations to filter
    Output:
        data_filtered = filtered data by sigma
    '''
    data_filtered = []

    for b, beatnote in enumerate(data):

        min_f = np.nanmean(beatnote) - sigma * np.nanstd(beatnote)
        max_f = np.nanmean(beatnote) + sigma * np.nanstd(beatnote)

        f_filtered = []
        f_excluded = []
        xf_excluded = []
        xf_filtered = []
        for i, j in enumerate(beatnote):
            if j > min_f and j < max_f:
                f_filtered.append(j)
                xf_filtered.append(i)
            else:
                f_excluded.append(j)
                xf_excluded.append(i)
        data_filtered.append([ xf_filtered, f_filtered ])
    return data_filtered

def select_higher_beatnotes(x, frequencies, beatnote_index, sigma, target_dir):
    beat_deriv = [np.gradient(f, 1) for f in frequencies]
    print(len(beat_deriv))

    selected_beats = [[] for _ in range(len(frequencies))]
    selected_times = [[] for _ in range(min(x), max(x))]

    print('Standard deviation = ', np.nanstd(beat_deriv)*10**-6,' MHz')
    threshold = np.nanstd(beat_deriv) * sigma

    for i in range(len(frequencies)):
        n_frames  = len(beat_deriv[i])

        for j,b in enumerate(beat_deriv[i]):
            if b >= threshold or b<= - threshold:
                selected_beats[i].append(b)
                selected_times[i].append(x[j])

        n_selected = len(selected_beats[i])    
        n_selected = len(selected_beats[i])
        #print(n_selected, 'beats selected for beat note ', i)
        
    if beatnote_index is None:
        tools.xy_plot([x, tools.convert_list(beat_deriv, 10**-6)], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = r'$\Delta$f (MHz)', title = 'Derivative of {} beat note frequencies over {} frames'.format(i+1, n_frames), box = False, save = False, target_dir = target_dir)

        #tools.xy_plot([selected_times[i], [selected_beats[i]]], type='beat_timeline', label_variable = ['Beatnote '+str(i+1)], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = r'$\Delta \nu$ (MHz)', title = r'Frequency change $\Delta \nu$', box = False, save = True, target_dir = target_dir)

        return selected_times, selected_beats
    else:
        tools.xy_plot([x, [tools.convert_list(beat_deriv, 10**-6)[beatnote_index-1]]], type='beat_timeline', label_variable = ['Beatnote '+str(beatnote_index)], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = r'$\Delta$f (MHz)', title = 'Derivative of beat note {} frequencies over {} frames'.format(beatnote_index, n_frames), box = False, save = False, target_dir = target_dir)
        tools.xy_plot([selected_times[beatnote_index-1], [selected_beats[beatnote_index-1]]], type='beat_timeline', label_variable = ['Beatnote '+str(beatnote_index)], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = r'$\Delta \nu$ (MHz)', title = r'Frequency change $\Delta \nu$', box = False, save = True, target_dir = target_dir)

        return selected_times[beatnote_index-1], selected_beats[beatnote_index-1]

'''CLASSIFY'''
def combine_fit_parameters(target_dir):
    frame_list = []
    filenames = mac_natsorted(os.listdir(target_dir))
    filenames = [f for f in filenames if '._' not in f]
    
    for f in filenames:
        if '_fit_parameters' in f:
            print('Available files to combine: ',f)
            frame = f.strip('_fit_parameters.npy')
            frame = frame.replace('[', '').replace(']', '')
            fl = frame.split(', ')
            frame_list.append([int(fl[0]), int(fl[1])])
            
    tempfiles= []
    
    frame_no = 0
    for i, n_frame in enumerate(frame_list):
        tf = np.load(target_dir+str(n_frame)+'_fit_parameters.npy', allow_pickle=True)
        
        frame_no = frame_list[i][1] - frame_list[i][0]
        
        a, f, w, fr = tf
        frame_indeces = np.arange(frame_list[i][0], frame_list[i][1]).tolist()
        tf = frame_indeces, a, f, w, fr
        
        print(n_frame)
        
        if i == 0:
            tempfiles = tf
        else:
            frame_indeces = np.arange(frame_list[i][0], frame_list[i][1]).tolist()
            tempfiles[0].extend(frame_indeces)
            for j in range(1,len(tf)):

                if j >= len(tf)-1:
                    tempfiles[j].append([np.nan] * (len(tempfiles[0])))
                   
                for k in range(len(tf[j])):
                    
                    if k+1 > len(tempfiles[j]):                         
                        tempfiles[j].append(tf[j][k])
                            
                    if j >= len(tf)-1:
                        tempfiles[j].append(tf[j][k])
    
                    else:
                        tempfiles[j][k] += tf[j][k]
    
    xframes, amplitudes, frequencies, widths, fits = tempfiles    
    amplitudes = [list(tpl) for tpl in zip(*zip_longest(*amplitudes, fillvalue = np.nan))]
    frequencies = [list(tpl) for tpl in zip(*zip_longest(*frequencies, fillvalue = np.nan))]
    widths = [list(tpl) for tpl in zip(*zip_longest(*widths, fillvalue = np.nan))]
    fits = [list(tpl) for tpl in zip(*zip_longest(*fits, fillvalue = np.nan))]
    np.save(target_dir+"fit_parameters.npy", np.array([xframes, amplitudes, frequencies, widths, fits], dtype=object)) 

