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

''''''''''''
        
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

   

def signal_track(frequency, signal_list, threshold, fstart, fend):

    coordinate_list = []
    f_diff = []
    highf = []
    high = []
    hight = []
    for j, s in enumerate(signal_list):
        for i in range(len(s)):
            if s[i] > threshold and frequency[i]>fstart and frequency[i]<fend:
                high.append(s[i])
                highf.append(frequency[i])
                hight.append(j)
                
                coordinate_list.append([j,frequency[i]])
    tools.xy_plot([[hight, highf]], aspect = 0.5, yerror = None, x_label = 'Frames', y_label = 'Frequency (Mhz)', title = 'Signal Intensity Track', box = False, save = False, target_dir = 'output/')            
    # plt.plot(hight, highf)
    # plt.ylim(fstart,fend)
    plt.show()
    
    return coordinate_list


def lorentzian_lineshape(x, a, x0, gam, offset):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2) + offset

def fit_lineshape(lineshape, x_time, y_data, pguess, label = 'fit'):
    popt, pcov = curve_fit(lineshape, x_time, y_data, p0 = pguess)

    return popt
    
def auto_lorentz(xdata, ydata):

    offset_guess = min(ydata)
    a_guess = max(ydata)
    x0_guess = np.argmax(ydata)
    gamma_guess = np.std(ydata)
    guess = [a_guess, xdata[x0_guess], gamma_guess, offset_guess]

    output = fit_lineshape(lorentzian_lineshape, xdata, ydata, pguess = guess)

    return output
@numba.jit
def odr_lorentzian_lineshape(params, x):
    a, x0, gam, offset = params
    return a * gam**2 / ( gam**2 + ( x - x0 )**2) + offset

@numba.jit
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

@numba.jit
def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        a = params[i]
        x0 = params[i+1]
        gam = params[i+2]
        offset = params[i+3]
        y = y + a * gam**2 / ( gam**2 + ( x - x0 )**2) + offset
    return y

@numba.jit
def odr_func(params, x):
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        a = params[i]
        x0 = params[i+1]
        gam = params[i+2]
        offset = params[i+3]
        y = y + a * gam**2 / ( gam**2 + ( x - x0 )**2) + offset
    return y

@numba.jit
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

def get_local_domain(peak_indices, window, interval, peak_values = None):
    print(peak_indices)
    
    deltas = []
    
    if len(peak_indices) > 0:
        n_peaks = len(peak_indices)
        nwindow = len(window)/n_peaks
        npwindow = int(nwindow/4)
        # print('window domain : ', npwindow)
        
        
        for gpeak in peak_indices:
            print('Peak ', gpeak)
            peak = gpeak - interval[0]
            print('Local peak ', peak)
            
            if peak - npwindow < 0:
                wmin = 0
            else:
                wmin = peak - npwindow
            if peak + npwindow >= len(window):
                wmax = len(window)
            else:
                wmax = peak + npwindow
            print(wmin, wmax)
            peak_window = window[ wmin : wmax ]
            
            # plt.plot(peak_window)
            # plt.show()
            
            # print('Local Domain: ', len(peak_window), peak_window)
            local_max = max(peak_window)
            local_min = min(peak_window)       
            delta = local_max - local_min
            
            print('max, min, delta = ', local_max, local_min, delta)
            
            deltas.append(delta)
            
            # plt.plot(peak_window)
            # plt.show()
    # delta_list = [peak_indices, deltas]
    return deltas
            

def multi_lorentz_fit(x, data, maxima, minima, minfit, plot):
    
    # plt.plot(x, data, color='black', linestyle='--')

    
    local = get_local_indices(maxima, len(data))
    # print('Local Indices: ',local)
    
    
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
        # guess += a, x0+min(local[i]), gam, off
        guess += a, x0, gam, off
            
        
        # plt.plot(x[min(local[i]):max(local[i])], data[min(local[i]):max(local[i])])
        # plt.plot(x[min(local[i]):max(local[i])], odr_lorentzian_lineshape(output, x[min(local[i]):max(local[i])]))
        # plt.show()
    
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
    fit = odr_func(popt, x)

    # popt, pcov = curve_fit(func, x, data, p0=guess)
    # fit = func(x, *popt)
    if plot is True:
        x = [x/10**6 for x in x]
        tools.xy_plot([[x, data]], fit = [fit], label_variable = None, aspect = 0.5, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Frame-by-frame Multi-Lorentzian Fit', box = False, save = True, target_dir = 'output/')

    print('Fit parameters [a, x0, gam, off]: ',popt)
    
    
    return amplitudes, frequencies, widths, fit

# @numba.jit
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

def remove_failed_fits(amplitudes, frequencies, widths, ydata, height):
    amplitudes2=[[] for _ in range(len(amplitudes))]
    frequencies2=[[] for _ in range(len(amplitudes))]
    widths2=[[] for _ in range(len(amplitudes))]
    frames2=[[] for _ in range(len(amplitudes))]
    
    frame_outlier=[[] for _ in range(len(amplitudes))]
    
    for i in range(len(amplitudes)):
        for j, a in enumerate(amplitudes[i]):
            if a<= max(ydata[0])*2 and a>height and widths[i][j] > -200:
                amplitudes2[i].append(a)
                frequencies2[i].append(frequencies[i][j])
                widths2[i].append(widths[i][j])
                frames2[i].append(j)
                
            else:
                print('Check outlier frame ',j)
                amplitudes2[i].append(np.nan)
                frequencies2[i].append(np.nan)
                widths2[i].append(np.nan)
                frames2[i].append(np.nan)
                
                frame_outlier[i].append(j)
                
                # t = np.arange(len(frequencies[i]))
    
    # tools.xy_plot([[t, frequencies[i]]], type='beat_timeline', label_variable = None, aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = 'Outlier Beat Frequency {} in frame {}'.format(i, j), box = False, save = False, target_dir = 'output/')
        print(len(frame_outlier[i]), ' outliers for beat note ',i)
    return amplitudes2, frequencies2, widths2, frames2, frame_outlier

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

def select_higher_beatnotes2(x, frequencies, sigma, target_dir):
    beat_deriv = [np.gradient(f, 1) for f in frequencies]
    print(len(beat_deriv))

    selected_beats = [[] for _ in range(len(frequencies))]
    selected_times = [[] for _ in range(min(x), max(x))]

    print('Standard deviation = ', np.nanstd(beat_deriv)*10**-6,' MHz')
    threshold = np.nanstd(beat_deriv) * sigma
    
    for i in range(len(frequencies)):
        n_frames  = len(beat_deriv[i])

        tools.xy_plot([[x, beat_deriv[i]]], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = r'$\Delta$f (MHz)', title = 'Derivative of beat note {} frequency over {} frames'.format(i, n_frames), box = False, save = False, target_dir = target_dir)
        
        for j,b in enumerate(beat_deriv[i]):
            if b >= threshold or b<= - threshold:
                selected_beats[i].append(b)
                selected_times[i].append(x[j])
    
        n_selected = len(selected_beats[i])        
        
        print(n_selected, 'beats selected for beat note ', i)
        # tools.xy_plot([[selected_times[i], selected_beats[i]]], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = r'Selected $\Delta$ f for beat note {}'.format(i), box = False, save = False, target_dir = 'output/')

        tools.xy_plot([[selected_times[i], selected_beats[i]]], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = r'$\Delta \nu$ (MHz)', title = r'Frequency change $\Delta \nu$', box = False, save = True, target_dir = target_dir)

    return selected_times, selected_beats

def select_higher_beats2(x, frequencies, threshold, target_dir):
    beat_deriv = np.gradient(frequencies, 1)
    print(len(beat_deriv))

    selected_beats = []
    selected_times = []
    
    if threshold is None:
        print('Standard deviation = ', np.nanstd(beat_deriv))
        threshold = np.nanstd(beat_deriv)*3
    
    tools.xy_plot([[x, beat_deriv]], aspect = 0.8, yerror = None, x_label = 'Time', y_label = 'Frequency (MHz)', title = 'Derivative', box = False, save = True, target_dir = target_dir)

    for j,b in enumerate(beat_deriv):
        if b >= threshold or b<= - threshold:
            selected_beats.append(b)
            selected_times.append(x[j])

    n_selected = len(selected_beats)        
        
    print(n_selected, 'beats selected')
    # tools.xy_plot([[selected_times[i], selected_beats[i]]], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = r'Selected $\Delta$ f for beat note {}'.format(i), box = False, save = False, target_dir = 'output/')

    tools.xy_plot([[selected_times, selected_beats]], type = 'beat_timeline', aspect = 0.8, yerror = None, x_label = 'Time', y_label = 'Frequency (MHz)', title = 'Selected derivatives', box = False, save = True, target_dir = target_dir)

    return selected_times, selected_beats


def cross_corr_plot(lags, ccf, target_dir):
    fig, ax =plt.subplots(figsize=(9, 6))
    ax.plot(lags, ccf)

    ax.axvline(x = 0, color = 'black', lw = 1)
    ax.axhline(y = 0, color = 'black', lw = 1)
    ax.axhline(y = np.max(ccf), color = 'red', lw = 1, linestyle='--', label = 'highest +/- correlation')
    ax.axhline(y = np.min(ccf), color = 'red', lw = 1, linestyle='--')
    ax.set(ylim = [-1, 1])
    ax.set_title('Cross Correlation between Two Signals')
    ax.set_ylabel('Correlation Coefficients')
    ax.set_xlabel('Time Lags')
    plt.legend()
    plt.savefig(target_dir+'cross_corr_plot.png')
    plt.show()



def data2selected(xdatatest, ydatatest, xstep, ystep, xspike, yspike, interval, save = False, target_dir = None):
    xstep = [xdatatest[f] for f in xstep]
    xspike = [xdatatest[f] for f in xspike]

    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'$\lambda$ (nm)', color='black')
    # plt.ylim(-3,-1.5)
    ax1.plot(xdatatest[interval[0]: interval[1]], ydatatest[interval[0]: interval[1]], color='grey', linewidth = 0.5, label = 'Data')
    ax1.tick_params(axis='y', labelcolor='black')
    # plt.ylim(970.02454, 970.02466)
    plt.xlim(xdatatest[interval[0]], xdatatest[interval[1]])
    ax1.legend()
    ax2 = ax1.twinx()
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'$\Delta \lambda$ (nm)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.ylim(-2*10**-5, 2*10**-5)
    
    selected_list = [ [xstep, ystep] , [xspike, yspike]]
    colors = ['tab:blue', 'tab:orange']
    labels = [r'$\Delta \lambda$ Step', r'$\Delta \lambda$ Spike']
    for p, pair in enumerate(selected_list):
        posxstep = []
        negxstep = []
        pos_step = []
        neg_step = []   
        for i, j in zip(pair[0], pair[1]):
            if j>0:
                pos_step.append(j)
                posxstep.append(i)
            if j<0:
                neg_step.append(j)
                negxstep.append(i)  
    
        ax2.scatter(posxstep, pos_step, color=colors[p], label = labels[p], marker = '^')
        ax2.scatter(negxstep, neg_step, color=colors[p], marker = 'v')

    
    
    fig.tight_layout()
    title = 'Selected Step, Spike Data'
    # plt.title(title)
    ax2.legend(loc='lower right')
    if save is True:
        plt.savefig(target_dir+str(title.replace(" ", "_"))+str(interval)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=1200)
    plt.show()
    
def convolve(datax, datay, interval, avg, filter_len = 20, save = False, target_dir = None):
    # avg = np.average(datay)
    # datay -= avg
    steps = np.hstack([np.ones(filter_len), -1*np.ones(filter_len)])
    convolution = np.convolve(datay, steps, mode='valid')
    
    globalMinIndex = np.argmin(convolution)+filter_len-1
    
    datay += avg
    
    fig, ax = plt.subplots()
    # ax = plt.gca()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\lambda$ (nm)')
    ax.plot(datax, datay, color = 'grey', linewidth = 0.5, label = 'data')
    ax.legend(loc = 'upper left')
    
    ax2 = ax.twinx()
    ax2.set_ylabel('Convolution', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    x = np.linspace(min(datax), max(datax), len(convolution))
    # ax2.plot(range(filter_len-1, len(datay)-filter_len),convolution/10, c="tab:blue", alpha=0.5, label = 'convolution')
    ax2.plot(x, convolution/10, c="tab:blue", alpha=1, label = 'convolution')
    ax2.legend(loc='upper center')
    
    fig.tight_layout()
    title = 'Signal Convolution'
    # plt.title(title)
    if save is True:
        plt.savefig(target_dir+str(title.replace(" ", "_"))+str(interval)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=1200)
    plt.show()
    return convolution

def convolve2(datax, datay, interval, avg, filter_len = 20, save = False, target_dir = None):
    # avg = np.average(datay)
    # datay -= avg
    # steps = np.hstack([np.ones(filter_len), -1*np.ones(filter_len)])
    # spikes = np.hstack([ -1*np.ones(filter_len), 1, -1*np.ones(filter_len-1)])
    # spikes = np.hstack([ np.zeros(filter_len-1), 1,-1, np.zeros(filter_len-1)])
    spikes = np.hstack([ np.zeros(filter_len), 1, np.zeros(filter_len-1)])
    
    
    convolution = np.convolve(datay, spikes, mode='valid')
    
    globalMinIndex = np.argmin(convolution)+filter_len-1
    
    datay += avg
    
    fig, ax = plt.subplots()
    # ax = plt.gca()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\lambda$ (nm)')
    ax.plot(datax, datay, color = 'grey', linewidth = 0.5, label = 'data')
    ax.legend(loc = 'upper left')
    
    ax2 = ax.twinx()
    ax2.set_ylabel('Convolution', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    x = np.linspace(min(datax), max(datax), len(convolution))
    # ax2.plot(range(filter_len-1, len(datay)-filter_len),convolution/10, c="tab:blue", alpha=0.5, label = 'convolution')
    ax2.plot(x, convolution/10, c="tab:blue", alpha=1, label = 'convolution')
    ax2.legend(loc='upper center')
    
    fig.tight_layout()
    title = 'Signal Convolution'
    # plt.title(title)
    if save is True:
        plt.savefig(target_dir+str(title.replace(" ", "_"))+str(interval)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=1200)
    plt.show()
    return convolution

def double_convolution_plot(x1, y1, y2, interval, steps_indices, steps, save = False, target_dir = None):
    fig, ax = plt.subplots()        
    # ax = plt.gca()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\lambda$ (nm)')
    ax.plot(x1, y1, color = 'grey', linewidth = 0.5, label = 'data')
    ax.legend(loc = 'upper left')
    
    ax2 = ax.twinx()
    ax2.set_ylabel('Convolution', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    x = list(np.linspace(min(x1), max(x1), len(y2)))
    if steps and steps_indices is not None:
        ax2.scatter(x[steps_indices[0] - interval[0]], steps[0]/10, marker = '.', color = 'tab:orange', label = 'steps')
        for j, step_indxs in enumerate(steps_indices):
            # ax2.plot((x[step_indxs - interval[0]], x[step_indxs - interval[0]]), (steps[j]/10, 0), color = 'tab:orange')
            ax2.scatter(x[step_indxs - interval[0]], steps[j]/10, marker = '.', color = 'tab:orange')
    # ax2.plot(range(filter_len-1, len(datay)-filter_len),convolution/10, c="tab:blue", alpha=0.5, label = 'convolution')
    ax2.plot(x, y2/10, c="tab:blue", alpha=1, label = 'convolution')
    ax2.legend(loc='upper center')
    
    fig.tight_layout()
    title = 'Signal Convolution'
    # plt.title(title)
    if save is True:
        plt.savefig(target_dir+str(title.replace(" ", "_"))+str(interval)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=1200)
    plt.show()

def double_convolution_plot2(n, x1, y1, x2, y2, interval, x_lim = None, save = False, target_dir = None):
    fig, ax = plt.subplots()        
    # ax = plt.gca()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\lambda$ (nm)')
    ax.plot(x1, y1, color = 'grey', linewidth = 0.5, label = 'data')
    ax.legend(loc = 'upper left')
    if x_lim is not None:
        plt.xlim(x1[x_lim[0]], x1[x_lim[1]])
        
        plt.ylim(y1[x_lim[0]], y1[x_lim[1]])
    
    ax2 = ax.twinx()
    ax2.set_ylabel('Step', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    x = [x1[i] for i in x2]

    ax2.scatter(x, y2, c="tab:blue", alpha=1, marker = '.', label = 'steps')
    ax2.legend(loc='upper center')
    # if x_lim is not None:
    #     plt.xlim(x1[x_lim[0]], x1[x_lim[1]])
    
    fig.tight_layout()
    title = 'Signal Convolution'
    # plt.title(title)
    if save is True:
        plt.savefig(target_dir+str(title.replace(" ", "_"))+str(interval)+str(n)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=1200)
    plt.show()
 
def double_convolution_plot3(n, x1, y1, x2, y2, interval, steps_indices, steps, x_lim = None, save = False, target_dir = None):
    fig, ax = plt.subplots()        
    # ax = plt.gca()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\lambda$ (nm)')
    ax.plot(x1, y1, color = 'grey', linewidth = 0.5, label = 'data')
    ax.legend(loc = 'upper left')
    if x_lim is not None:
        plt.xlim(x1[x_lim[0]], x1[x_lim[1]])
        
        plt.ylim(y1[x_lim[0]], y1[x_lim[1]])
    
    ax2 = ax.twinx()
    ax2.set_ylabel('Convolution', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # x = list(np.linspace(min(x1), max(x1), len(y2)))
    # x = [x1[i] for i in x2]
    # if steps and steps_indices is not None:
    #     for x22, y22 in zip(x2, y2):
    #         ax2.plot((x1[x22], x1[x22]), (y22, 0), 'tab:orange')
    # # ax2.plot(range(filter_len-1, len(datay)-filter_len),convolution/10, c="tab:blue", alpha=0.5, label = 'convolution')
    # x = list(np.linspace(min(x1), max(x1), len(y2)))
    # ax2.plot(x, np.array(y2), c="tab:blue", alpha=1, label = 'convolution')
    # ax2.legend(loc='upper center')
    
    x = list(np.linspace(x1[interval[0]], x1[interval[1]], len(y2)))
    # x = list(np.linspace(x1[interval[0]], x1[interval[1]], len(y2)))
    print(x)
    if steps and steps_indices is not None:
        for j, step_indxs in enumerate(steps_indices):
            # ax2.plot((x[step_indxs], x[step_indxs]), (steps[j], 0), 'tab:orange')
            ax2.plot((x[step_indxs - interval[0]], x[step_indxs- interval[0]]), (steps[j], 0), 'tab:orange', zorder=1)
            # ax2.plot((x[step_indxs], x[step_indxs]), (steps[j], 0), 'tab:orange')
    # ax2.plot(range(filter_len-1, len(datay)-filter_len),convolution/10, c="tab:blue", alpha=0.5, label = 'convolution')
    # ax2.plot(x1[interval[0]:interval[1]], np.array(y2)/10, c="tab:blue", alpha=1, label = 'convolution')
    ax2.plot(x, np.array(y2)/10, c="tab:blue", alpha=0.5, label = 'convolution', zorder=0)
    ax2.plot((x1[x_lim[0]], x1[x_lim[1]]), (0, 0), color = 'black', linestyle = '--', linewidth = 0.5)
    
    fig.tight_layout()
    title = 'Signal Convolution'
    # plt.title(title)
    if save is True:
        plt.savefig(target_dir+str(n)+'_'+str(title.replace(" ", "_"))+str(interval)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=300)
    plt.show()   
 
def step_finder2(interval, time, datay, window_size, type, minfit = True, distance = 50, width = 25, height = None, dev = 1, plot = True, target_dir = None):
    
    # i = 0
    maximas = []
    minimas = []

    steps_indices = []
    steps_excluded = []
    
    convolutions = []
    
    delta_lambda = []
    
    window_no = int(len(datay) / window_size)
    print(window_no,' windows to sample')
    
    datax = np.linspace(interval[0], interval[1], len(datay))
    
    dary_total = []
    steps = []
    
    avg = np.average(datay)
    datayy = datay - avg

    for i in range(0, window_no):
        print('window ', i)
        
        x_min = interval[0] + (window_size*i)
        x_max = x_min + window_size

        window = datay[i*window_size : (i+1)*window_size]
        windowx = datax[i*window_size : (i+1)*window_size]
        dary_total.extend(window)
        
        windoww = datayy[i*window_size : (i+1)*window_size]

        convolution = convolve(windowx, windoww, interval, avg, filter_len = 20, save = False, target_dir = target_dir)
        convolutions.extend(convolution)
        
        '''
        Get max and min indices of peak convolution
        '''
        # windowx = np.linspace(x_min, x_max, len(convolution))
        # sigma_step = np.std(convolution)
        # if height is None:
        #     height = sigma_step*dev
        
        # maxima, minima = tools.peak_finder(windowx, convolution, minfit = minfit, threshold=None, width=width, height=height, distance=distance, prominence=None, plot=False)
        # xmaxima = [int(windowx[m]) for m in maxima]
        # xminima = [int(windowx[m]) for m in minima]

        # maximas.extend(list(xmaxima))
        # minimas.extend(list(xminima))

        # step_indices = [*xmaxima, *xminima]
        # xstep_indices = [*maxima, *minima]
        
        # step_excluded = [d/10 for d in convolution if d not in maxima and d not in minima]
        # steps_excluded.extend(step_excluded[:-1])
        
        # deltamax = get_local_domain(xmaxima, datay, interval)
        # print('deltamax ', deltamax)
        # delta_lambda.extend(list(deltamax))
        # deltamin = get_local_domain(xminima, datay, interval)
        # deltamin = [-i for i in deltamin]
        # print('deltamin ', deltamin)
        # delta_lambda.extend(list(deltamin))
        

        if plot is True:
            double_convolution_plot(windowx, window, np.array(convolution)/10, interval, steps_indices = None, steps = None, save = False, target_dir = target_dir)
            # plt.plot(windowx, convolution/10)
            # plt.show()
    convolutions -= np.nanmean(convolutions)
    sigma_step = np.std(convolutions)
    if height is None:
        height = sigma_step*dev
    maxima, minima = tools.peak_finder(datax, convolutions, minfit = minfit, threshold=None, width=width, height=height, distance=distance, prominence=None, plot=False)
    maximas = [int(datax[m]) for m in maxima]
    minimas = [int(datax[m]) for m in minima]

    steps_indices = [*maximas, *minimas]
    xstep_indices = [*maxima, *minima]
    print(len(steps_indices), steps_indices)
    print(len(xstep_indices), xstep_indices)
    print(len(convolutions))
    
    steps = [convolutions[xs]/10 for xs in xstep_indices]
    steps_excluded = [d/10 for d in convolution if d not in maxima and d not in minima]
    print(len(steps), steps)
    
    double_convolution_plot(time, datay, np.array(convolutions)/10, interval, steps_indices, steps, save = True, target_dir = target_dir)
    
    
    return maximas, minimas, steps_indices, steps, steps_excluded, delta_lambda, convolutions

def spike_finder(interval, time, datay, window_size, type, minfit = True, distance = 50, width = 25, height = None, dev = 1, plot = True, target_dir = None):
    
    
    # i = 0
    maximas = []
    minimas = []

    steps_indices = []
    steps_excluded = []
    
    convolutions = []
    
    delta_lambda = []
    
    window_no = int(len(datay) / window_size)
    print(window_no,' windows to sample')
    
    datax = np.linspace(interval[0], interval[1], len(datay))
    
    dary_total = []
    steps = []
    
    avg = np.average(datay)
    datayy = datay - avg

    for i in range(0, window_no):
        print('window ', i)
        
        x_min = interval[0] + (window_size*i)
        x_max = x_min + window_size

        window = datay[i*window_size : (i+1)*window_size]
        windowx = datax[i*window_size : (i+1)*window_size]
        dary_total.extend(window)
        
        windoww = datayy[i*window_size : (i+1)*window_size]

        convolution = convolve2(windowx, windoww, interval, avg, filter_len = 20, save = False, target_dir = target_dir)
        convolutions.extend(convolution)
        

        if plot is True:
            double_convolution_plot(windowx, window, np.array(convolution)/10, interval, steps_indices = None, steps = None, save = False, target_dir = target_dir)
            # plt.plot(windowx, convolution/10)
            # plt.show()
    convolutions -= np.nanmean(convolutions)
    sigma_step = np.std(convolutions)
    if height is None:
        height = sigma_step*dev
    maxima, minima = tools.peak_finder(datax, convolutions, minfit = minfit, threshold=None, width=width, height=height, distance=distance, prominence=None, plot=False)
    maximas = [int(datax[m]) for m in maxima]
    minimas = [int(datax[m]) for m in minima]

    steps_indices = [*maximas, *minimas]
    xstep_indices = [*maxima, *minima]
    print(len(steps_indices), steps_indices)
    print(len(xstep_indices), xstep_indices)
    print(len(convolutions))
    
    steps = [convolutions[xs]/10 for xs in xstep_indices]
    steps_excluded = [d/10 for d in convolution if d not in maxima and d not in minima]
    print(len(steps), steps)
    
    double_convolution_plot(time, datay, np.array(convolutions)/10, interval, steps_indices, steps, save = True, target_dir = target_dir)
    
    
    return maximas, minimas, steps_indices, steps, steps_excluded, delta_lambda, convolutions


'''RESONANCE PIPELINE'''
def background_correction(x, y, window_size):
    baseline = np.nanmean(y)
    print('Baseline = ', baseline)
        
    i = 0
    moving_averages = []
    y_corr = []
    

    while i < len(y) - window_size + 1:

        window = y[i : i + window_size]
        window_average = sum(window) / window_size
        
        moving_averages.append(window_average)
        y_corr.append(y[i] - window_average)

        i += 1

    y_corr = [(k)+baseline for k in y_corr]

    return moving_averages, y_corr


def convolvei(datax, datay, type, interval, avg, filter_len = 20, save = False, target_dir = None):
    if type == 'steps':
        steps = np.hstack([np.ones(filter_len), -1*np.ones(filter_len)])
    if type == 'spikes':
        spikes = np.hstack([ np.zeros(filter_len), 1, np.zeros(filter_len-1)])
    
    convolution = np.convolve(datay, spikes, mode='valid')
    globalMinIndex = np.argmin(convolution)+filter_len-1
    
    datay += avg
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\lambda$ (nm)')
    ax.plot(datax, datay, color = 'grey', linewidth = 0.5, label = 'data')
    ax.legend(loc = 'upper left')
    
    ax2 = ax.twinx()
    ax2.set_ylabel('Convolution', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    x = np.linspace(min(datax), max(datax), len(convolution))
    # ax2.plot(range(filter_len-1, len(datay)-filter_len),convolution/10, c="tab:blue", alpha=0.5, label = 'convolution')
    ax2.plot(x, convolution/10, c="tab:blue", alpha=1, label = 'convolution')
    ax2.legend(loc='upper center')
    
    fig.tight_layout()
    title = str(type)+' Convolution'
    plt.title(title)
    if save is True:
        plt.savefig(target_dir+str(title.replace(" ", "_"))+str(interval)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=1200)
    plt.show()
    return convolution

def double_convolution_plot(x1, y1, y2, interval, steps_indices, steps, save = False, target_dir = None):
    fig, ax = plt.subplots()        
    # ax = plt.gca()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\lambda$ (nm)')
    ax.plot(x1, y1, color = 'grey', linewidth = 0.5, label = 'data')
    ax.legend(loc = 'upper left')
    
    ax2 = ax.twinx()
    ax2.set_ylabel('Convolution', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    x = list(np.linspace(min(x1), max(x1), len(y2)))
    if steps and steps_indices is not None:
        ax2.scatter(x[steps_indices[0] - interval[0]], steps[0]/10, marker = '.', color = 'tab:orange', label = 'steps')
        for j, step_indxs in enumerate(steps_indices):
            # ax2.plot((x[step_indxs - interval[0]], x[step_indxs - interval[0]]), (steps[j]/10, 0), color = 'tab:orange')
            ax2.scatter(x[step_indxs - interval[0]], steps[j]/10, marker = '.', color = 'tab:orange')
    # ax2.plot(range(filter_len-1, len(datay)-filter_len),convolution/10, c="tab:blue", alpha=0.5, label = 'convolution')
    ax2.plot(x, y2/10, c="tab:blue", alpha=1, label = 'convolution')
    ax2.legend(loc='upper center')
    
    fig.tight_layout()
    title = 'Signal Convolution'
    # plt.title(title)
    if save is True:
        plt.savefig(target_dir+str(title.replace(" ", "_"))+str(interval)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=1200)
    plt.show()

def signal_finder(interval, time, datay, type, window_size, minfit = True, distance = 50, width = 25, height = None, dev = 1, plot = True, target_dir = None):
    
    maximas = []
    minimas = []
    steps_indices = []
    steps_excluded = []
    convolutions = []
    delta_lambda = []
    
    window_no = int(len(datay) / window_size)
    print(window_no,' windows to sample')
    
    datax = np.linspace(interval[0], interval[1], len(datay))
    
    dary_total = []
    steps = []
    
    avg = np.average(datay)
    datayy = datay - avg

    for i in range(0, window_no):
        print('window ', i)
        
        x_min = interval[0] + (window_size*i)
        x_max = x_min + window_size

        window = datay[i*window_size : (i+1)*window_size]
        windowx = datax[i*window_size : (i+1)*window_size]
        dary_total.extend(window)
        
        windoww = datayy[i*window_size : (i+1)*window_size]

        convolution = convolvei(windowx, windoww, type, interval, avg, filter_len = 20, save = False, target_dir = target_dir)
        convolutions.extend(convolution)
        

        if plot is True:
            double_convolution_plot(windowx, window, np.array(convolution)/10, interval, steps_indices = None, steps = None, save = False, target_dir = target_dir)

    convolutions -= np.nanmean(convolutions)
    sigma_step = np.std(convolutions)
    if height is None:
        height = sigma_step*dev
    maxima, minima = tools.peak_finder(datax, convolutions, minfit = minfit, threshold=None, width=width, height=height, distance=distance, prominence=None, plot=False)
    maximas = [int(datax[m]) for m in maxima]
    minimas = [int(datax[m]) for m in minima]

    steps_indices = [*maximas, *minimas]
    xstep_indices = [*maxima, *minima]
    print(len(steps_indices), steps_indices)
    print(len(xstep_indices), xstep_indices)
    print(len(convolutions))
    
    steps = [convolutions[xs]/10 for xs in xstep_indices]
    steps_excluded = [d/10 for d in convolutions if d not in maxima and d not in minima]
    print(len(steps), steps)
    
    double_convolution_plot(time, datay, np.array(convolutions)/10, interval, steps_indices, steps, save = True, target_dir = target_dir)

    np.save(target_dir+str(type)+"_resonance_parameters.npy", np.array([maximas, minimas, steps_indices, steps, steps_excluded, delta_lambda, convolutions], dtype = object))

    return maximas, minimas, steps_indices, steps, steps_excluded, delta_lambda, convolutions