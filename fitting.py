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

import tools as tools
from itertools import zip_longest



def spectrum(frequency, signal_corr, crop, frame_crop=None, start=None, end=None, vmin=0, vmax=10, save=False, target_dir = 'output/'):
    fstart= frame_crop[0]
    fend= frame_crop[1]
    print(len(signal_corr[0]))
    signals = np.reshape(signal_corr[fstart:fend], (len(signal_corr[fstart:fend]), len(signal_corr[0]))).T
    frames = np.arange(crop[0], crop[1]-1)
    
    # datasets = [[frames, signals[0]]]
    # xy_plot(datasets, aspect = 0.8, title = 'Time Series of Signal Intensity at Frequency X', x_label = 'Frames', y_label = 'Intensity (dBm)')

    
    
    print('N_freq: ',len(frequency))
    print('N_frames: ',len(frames))
    print('N_signals: ',len(signals))
    
    fig, ax = plt.subplots(figsize = (15,9))

    ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
    ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    

    plt.pcolormesh(frames, frequency[tools.frequency_crop(start, frequency):tools.frequency_crop(end, frequency)], signals[tools.frequency_crop(start, frequency):tools.frequency_crop(end, frequency)],shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')
    
    plt.colorbar()

    ax.set_xlabel('Frames',fontsize=18,labelpad = 15)
    ax.set_ylabel('Frequency (MHz)',fontsize=18,labelpad = 15)
    
    if save:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        plt.savefig(target_dir+"spectrum_plot.png", bbox_inches='tight',pad_inches=0.0)
        plt.show()
    else:
        plt.show()
        
def spectrum2(frequency, signal_corr, vmin=0, vmax=10, save=False, target_dir = 'output/'):

    # signals = np.reshape(signal_corr, (len(signal_corr), len(signal_corr[0]))).T
    # signals = np.array(signal_corr).T
    # frames = np.arange(0,len(signals[0]))
    # print(frames)
    
    # datasets = [[frames, signals[0]]]
    # xy_plot(datasets, aspect = 0.8, title = 'Time Series of Signal Intensity at Frequency X', x_label = 'Frames', y_label = 'Intensity (dBm)')

    
    
    # print('N_freq: ',len(frequency))
    # print('N_frames: ',len(frames))
    # print('N_signals: ',len(signals))
    
    fig, ax = plt.subplots(figsize = (15,9))

    ax.tick_params(which = 'major', labelsize = 18, direction = 'in', length = 15, width = 1, bottom = True, top = True, left = True, right = True)
    ax.tick_params(which = 'minor', labelsize = 0, direction = 'in', length = 7, width = 1, bottom = True, top = True, left = True, right = True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    

    # plt.pcolormesh(frames, frequency[frequency_crop(start, frequency):frequency_crop(end, frequency)], signals[frequency_crop(start, frequency):frequency_crop(end, frequency)],shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')
    
    
    f, t, Sxx = signal.spectrogram(signal_corr, fs=5e9)

    plt.pcolormesh(t, f, Sxx, cmap='plasma')
    
    # plt.pcolormesh(frames, frequency, signals,shading='gouraud', vmin=vmin, vmax=vmax, cmap='inferno')

    plt.colorbar()

    ax.set_xlabel('Frames',fontsize=18,labelpad = 15)
    ax.set_ylabel('Frequency (MHz)',fontsize=18,labelpad = 15)
    
    if save:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        plt.savefig(target_dir+"spectrum_plot.png", bbox_inches='tight',pad_inches=0.0)
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
    gamma_guess = 0.05
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
        tools.xy_plot([[x, data]], fit = [fit], label_variable = None, aspect = 0.33, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Frame-by-frame Multi-Lorentzian Fit', box = False, save = False, target_dir = 'output/')

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
            
        # elif len(maxima) > maximals[0]:
        #     amplitudes.append([np.nan] * nn_frames)
        #     frequencies.append([np.nan] * nn_frames)
        #     widths.append([np.nan] * nn_frames)
            
        #     amplitude, frequency, wid, fit = multi_lorentz_fit(x_ma, data, maxima, minima, minfit, plot=plot)
        #     fits.append(fit)
            
        else:
            amplitude, frequency, wid, fit = multi_lorentz_fit(x_ma, data, maxima, minima, minfit, plot=plot)
            fits.append(fit)
        
        
        for m in range(len(maxima)):
            if m+1 > len(amplitudes):
                print(i)
                # amplitudes.append([])
                # frequencies.append([])
                # widths.append([])
                
                # amplitudes[m] = [np.nan] * nn_frames
                # frequencies[m] = [np.nan] * nn_frames
                # widths[m] = [np.nan] * nn_frames
                
                amplitudes.append([np.nan] * (i - len(amplitudes[m])))
                frequencies.append([np.nan] * (i - len(frequencies[m])))
                widths.append([np.nan] * (i - len(widths[m])))
                print(len(amplitudes[m]))
                
                # amplitudes[m][i] = amplitude[m]
                # frequencies[m][i] = frequency[m]
                # widths[m][i] = wid[m]
                
                amplitudes[m] += amplitude[m]
                frequencies[m] += frequency[m]
                widths[m] += wid[m]
            else:
                amplitudes[m] += amplitude[m]
                frequencies[m] += frequency[m]
                widths[m] += wid[m]
  
        
        # fits.append(fit)
    # for a, amp in enumerate(amplitudes):
    #     if len(amplitudes[a]) < len(amplitudes[0]):
    #         dif = int( len(amplitudes[0]) - len(amplitudes[a]))
    #         amplitudes[a].append([np.nan]*dif)
    
    amplitudes = [list(tpl) for tpl in zip(*zip_longest(*amplitudes, fillvalue = np.nan))]
    frequencies = [list(tpl) for tpl in zip(*zip_longest(*frequencies, fillvalue = np.nan))]
    widths = [list(tpl) for tpl in zip(*zip_longest(*widths, fillvalue = np.nan))]
    fits = [list(tpl) for tpl in zip(*zip_longest(*fits, fillvalue = np.nan))]
    
    np.save(target_dir+str(n_frames)+"_fit_parameters.npy", [amplitudes, frequencies, widths, fits])
    
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

def check_outliers(xdata, ydata, fits, n_avg, frame_outlier):
    for out in frame_outlier:
        yp = tools.moving_average(ydata[out], n_avg, factor = 40, floor=False)
        xp = tools.moving_average(xdata, n_avg, factor = 40, floor=False)
        plt.plot(xp, yp)
        plt.plot(xp, fits[out])
        plt.show()

def select_higher_beats(frequencies, threshold):
    beat_deriv = [np.gradient(f, 1) for f in frequencies]

    selected_beats = [[] for _ in range(len(frequencies))]
    selected_times = [[] for _ in range(len(frequencies))]
    
    for i in range(len(frequencies)):
        n_frames  = len(beat_deriv[i])
        t = np.arange(len(beat_deriv[i]))
        tools.xy_plot([[t, beat_deriv[i]]], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = r'$\Delta$f (MHz)', title = 'Derivative of beat note {} frequency over {} frames'.format(i, n_frames), box = False, save = False, target_dir = 'output/')
        
        for j,b in enumerate(beat_deriv[i]):
            if b >= threshold or b<= - threshold:
                selected_beats[i].append(b)
                selected_times[i].append(t[j])
    
        n_selected = len(selected_beats[i])          
        print(n_selected, 'beats selected for beat note ', i)
        tools.xy_plot([[selected_times[i], selected_beats[i]]], type='beat_timeline', label_variable = ['Beatnote '+str(i+1) for i in range(len(frequencies))], aspect = 0.33, yerror = None, x_label = 'Frame', y_label = 'Frequency (MHz)', title = r'Selected $\Delta$ f for beat note {}'.format(i), box = False, save = False, target_dir = 'output/')

    return selected_times, selected_beats