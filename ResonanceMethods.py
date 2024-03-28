#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:55:22 2024

@author: sabrinaperrenoud
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from scipy.optimize import curve_fit

import tools as tools

'''RESONANCE METHODS'''


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


def convolve(datax, datay, type, interval, avg, filter_len = 20, save = False, target_dir = None):
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

        convolution = convolve(windowx, windoww, type, interval, avg, filter_len = 20, save = False, target_dir = target_dir)
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