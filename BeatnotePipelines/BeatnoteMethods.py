#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import scipy.odr as odr
from scipy import stats
from scipy import signal
from itertools import zip_longest

import tools as tools
import fitting as fit

'''
BEATNOTE METHODS

Mid-level functions called on by BeatnotePipeline
'''


def process_dict(dictionary):
    '''
    Function: Classifies data in dictionary
    Input:
        dictionary = raw data dictionary from loaded data
    Output:
        data = dictionary of data for all .csv files
    '''
    
    filenames = dictionary.keys()
    print(filenames)
    
    data = {}
    
    for f in filenames:
        load = dictionary[f]
        print(load.keys())
        frequency = load['Frequency']
        cutoff = int(len(frequency)/2)
        frequency = np.array([float(f) for f in frequency])[:cutoff]
        
        crop = [0, len(load) - 1]

        signal_list = []
        for i in range(crop[1]-crop[0]):
            signal = load[str(i)]
                
            signal_array = np.array([float(s) for s in signal])[:cutoff]
            signal_list.append(signal_array)
                


        data[f] = [frequency, signal_list]
                              
    return data

def correct_background(dictionary, data_filename, background_filename):
    '''
    Function: Removes the average background from the data
    Input:
        dictionary = data files
        data_filename = string of data file to correct
        background_filename = string of bacground file for correction
    Output:
        data_corr = background corrected data
    '''
    filenames = dictionary.keys()
    print(filenames)
    
    bg_avg = 0

    background = []

    bg = dictionary[str(background_filename)]
    bgfrequency = bg[0]
    cutoff = int(len(bgfrequency)/2)
    bgfrequency = np.array([float(f) for f in bgfrequency])[:cutoff]
    background.append(bgfrequency)
    bgsignal_list = []
    for i in range(len(bg[1]) - 1):
        bgsignal = bg[1][i]
        bgsignal_array = np.array([float(s) for s in bgsignal])[:cutoff]
        bgsignal_list.append(bgsignal_array)
    background.append(bgsignal_list)
    
    bg_avg = tools.background_avg(background)
    
    load = dictionary[data_filename]
    frequency = load[0]
    cutoff = int(len(frequency)/2)
    frequency = np.array([float(f) for f in frequency])[:cutoff]
    
    crop = [0, len(load) - 1]

    signal_list = []
    signal_corr_list = []
    for i in range(crop[1]-crop[0]):
        signal = load[1][i]
            
        signal_array = np.array([float(s) for s in signal])[:cutoff]
        signal_list.append(signal_array)
        
        signal_corr = signal_array  - np.array(bg_avg)
        signal_corr_list.append(signal_corr)
            
    data_corr = [frequency, signal_corr_list]
        
    return data_corr

def spectrum(xdata, ydata, vmin, vmax, crop, save, target_dir):
    '''
    Function: Plots 3D spectrum of data
    Input:
        xdata = list of frames
        ydata = list of frequencies
        vmin = lowest intensity spectrum value
        vmax = highest intensity spectrum value
        crop = [,]/None to crop frequency range
        save = True/False
        target_dir = output directory for saving plot
    Output:
        plot of spectrum
    '''
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

def multi_lorentz_fit(x, data, maxima, minima, minfit, plot):
    '''
    Function: Fits each identified peak in signal with a lorentzian within local domain
    Input:
        x = list of frequencies
        data = list of intensities
        maxima = list of peak frequency position
        minima = list of peak minima frequency position
        minfit = True/False to include minima
        plot = True/False, set to False to speed up function
    Output:
        amplitudes, frequencies, widths, fit = fit parameters for combined function of all peaks
    '''
    local = fit.get_local_indices(maxima, len(data))

    amplitudes = [[] for _ in range(len(maxima))]
    frequencies = [[] for _ in range(len(maxima))]
    widths = [[] for _ in range(len(maxima))]
    
    
    guess = ()
    print(guess)

    for i in range(len(maxima)):
        print('Fitting ',i+1,'out of ', len(maxima),'peaks')
        # print(local[i])
        print('Domain to fit: ',min(local[i]),max(local[i]))
        
        output = fit.odr_auto_lorentz(x[min(local[i]):max(local[i])], data[min(local[i]):max(local[i])], maxima[i]-min(local[i]))
        print('OUTPUT',output)
        a, x0, gam, off = output

        
        frequencies[i].append(x[maxima[i]])
        amplitudes[i].append(a)
        widths[i].append(gam)
        guess += a, x0, gam, off

    
    if minfit is True:
        local2 = fit.get_local_indices(minima, len(data))
        
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
    
    popt, pcov = fit.odrfit(odr_func, x, data, initials = guess, param_mask = param_mask)
    lorentzfit = fit.odr_lorentz_func(popt, x)

    # popt, pcov = curve_fit(func, x, data, p0=guess)
    # fit = func(x, *popt)
    if plot is True:
        x = [x/10**6 for x in x]
        tools.xy_plot([[x, data]], fit = [lorentzfit], label_variable = None, aspect = 0.5, yerror = None, x_label = 'Frequency (MHz)', y_label = 'Intensity (dB)', title = 'Frame-by-frame Multi-Lorentzian Fit', box = False, save = True, target_dir = 'output/')

    print('Fit parameters [a, x0, gam, off]: ',popt)
    
    
    return amplitudes, frequencies, widths, lorentzfit

def cycle_fit(x, y, n_avg, n_frames, minfit, threshold, width, height, distance, prominence, target_dir, plot=False):
    '''
    Function: Fits beatnotes in each frame with a multi-lorentzian function
    Input:
        x = list of frequencies
        y = list of intensities
        n_avg = moving average, determines length of averaging window
        minfit = True/False to include fitting of minima
        threshold = minimum peak value
        width = minimum peak width
        height = minimum peak height
        distance = minimum distance between neighbouring peaks
        prominence = peak prominence, set to None
        target_dir = output directory for saving data/plots
        plot = True/False, set to False to speed up function
    Output:
        amplitudes, frequencies, widths, fits = fit parameters
        saves fit parameters in binary .npy file
    '''
    
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
            amplitude, frequency, wid, lorentzfit = nanlist, nanlist, nanlist, nanlist
            maxima = [np.nan] * len(amplitudes)
            fits.append(lorentzfit)

            
        else:
            amplitude, frequency, wid, lorentzfit = multi_lorentz_fit(x_ma, data, maxima, minima, minfit, plot=plot)
            fits.append(lorentzfit)
            
        
        
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
    '''
    Function: Selects frequency shifts above a certain number of standard deviations
    Input:
        x = list of times
        frequencies = list of frequencies
        beatnote_index = None/i, which beatnote to analyse
        sigma = threshold number of standard deviations
    Output:
        selected_beats = filtered beatnote frequency shifts
        selected_times = corresponding times
    '''
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
