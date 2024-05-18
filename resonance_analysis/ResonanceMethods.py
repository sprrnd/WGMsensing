#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:55:22 2024

@author: sabrinaperrenoud
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy import stats

import tools as tools

'''
RESONANCE METHODS

Mid-level functions called on by ResonancePipeline
'''


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
        convolution = np.convolve(datay, steps, mode='valid')
    if type == 'spikes':
        spikes = np.hstack([ np.zeros(filter_len), 1, -1, np.zeros(filter_len-2)])
        convolution = np.convolve(datay, spikes, mode='valid')
    
    globalMinIndex = np.argmin(convolution)+filter_len-1
    
    datay += avg
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Indices')
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

def double_convolution_plot(x1, y1, y2, interval, signal_indices, signal_heights, save = False, target_dir = None):
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
    if signal_heights and signal_indices is not None:
        ax2.scatter(x[signal_indices[0] - interval[0]], signal_heights[0]/10, marker = '.', color = 'tab:orange', label = 'signal')
        for j, step_indxs in enumerate(signal_indices):
            # ax2.plot((x[step_indxs - interval[0]], x[step_indxs - interval[0]]), (steps[j]/10, 0), color = 'tab:orange')
            ax2.scatter(x[step_indxs - interval[0]], signal_heights[j]/10, marker = '.', color = 'tab:orange')
    # ax2.plot(range(filter_len-1, len(datay)-filter_len),convolution/10, c="tab:blue", alpha=0.5, label = 'convolution')
    ax2.plot(x, y2/10, c="tab:blue", alpha=1, label = 'convolution')
    ax2.legend(loc='upper center')
    
    fig.tight_layout()
    title = 'Signal Convolution'
    # plt.title(title)
    if save is True:
        plt.savefig(target_dir+str(title.replace(" ", "_"))+str(interval)+"_plot.png", bbox_inches='tight',pad_inches=0.0, format = 'png', dpi=1200)
    plt.show()

def signal_finder(interval, time, datay, type, window_size, minfit = True, distance = None, width = None, height = None, dev = 3, plot = True, target_dir = None):
    
    maximas = []
    minimas = []
    signal_indices = []
    signal_heights = []
    signal_excluded = []
    convolutions = []
    
    window_no = int(len(datay) / window_size)
    print(window_no,' windows to sample')
    
    datax = np.linspace(interval[0], interval[1], len(datay))
    
    dary_total = []
    
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
            double_convolution_plot(windowx, window, np.array(convolution)/10, interval, signal_indices = None, signal_heights = None, save = False, target_dir = target_dir)

    convolutions -= np.nanmean(convolutions)
    sigma_step = np.std(convolutions)
    if height is None:
        height = sigma_step*dev
    maxima, minima = tools.peak_finder(datax, convolutions, minfit = minfit, threshold=None, width=width, height=height, distance=distance, prominence=None, plot=False)
    maximas = [int(datax[m]) for m in maxima]
    minimas = [int(datax[m]) for m in minima]

    signal_indices = [*maximas, *minimas]
    xsignal_indices = [*maxima, *minima]
    print(len(signal_indices), signal_indices)
    print(len(xsignal_indices), xsignal_indices)
    print(len(convolutions))
    
    signal_heights = [convolutions[xs]/10 for xs in xsignal_indices]
    signal_excluded = [d/10 for d in convolutions if d not in maxima and d not in minima]
    print(len(signal_heights), signal_heights)
    
    double_convolution_plot(time, datay, np.array(convolutions)/10, interval, signal_indices, signal_heights, save = True, target_dir = target_dir)

    np.save(target_dir+str(type)+"_resonance_parameters.npy", np.array([maximas, minimas, signal_indices, signal_heights, signal_excluded, convolutions], dtype = object))

    return maximas, minimas, signal_indices, signal_heights, signal_excluded, convolutions

def single_gaussian_fit(data, ind, save = False, target_dir = None):
    y,x,_= plt.hist(data, bins = 20, label='Data', color = 'lightgrey')
    x=(x[1:]+x[:-1])/2
    plt.show()

    mean1 = np.nanmean(x[ind:])
    sigma1 = np.nanstd(x[ind:])
    A1 = max(y[ind:])

    expected = (mean1, sigma1, A1)
    print('Expected: ', expected)
        
    try:
        params, cov = curve_fit(gauss, x, y, expected)
        print('Fit parameters = ', params)

        x_fit = np.linspace(-np.abs(x).max(), np.abs(x).max(), 500)
        gaussian = gauss(x_fit, *params[:3])
        
        plt.plot(x_fit, gaussian, color='tab:blue', lw=1, ls="--", label=r'Fit 1: $\mu ={},\sigma ={}$'.format(round(params[0], 7),round(params[1],8)))
        
        plt.xlim(min(x), max(x))
        plt.ylim(0, max(y)+1)
        plt.hist(data, bins = len(data), label='Data', color = 'lightgrey')
        plt.legend()
        plt.title('Single Gaussian Fit')
        plt.ylabel('n')
        plt.xlabel(r'$\Delta \lambda$ (nm)')
        if save is True:
            plt.savefig(target_dir+'single_gaussian_fit.png', dpi=1200)
        plt.show() 

    except RuntimeError:
        params = None
        gaussian = None
        print("Error - curve_fit failed")

    return params, gaussian

def get_counts(step_indices, steps, window, window_no):
    counts = []
    for f in range(window_no):
        selected = []
        for i, ind in enumerate(step_indices):
            # if ind in indeces:
            if ind > 0 + window*f and ind < window + window*f:
                print(i)
                selected.append(steps[i])
        
        c = len(selected)
        print(c, ' counts detected in window')
        counts.append(c)
    return counts

def poisson_fit_histogram(data):
    entries, bin_edges, patches = plt.hist(data, bins=max(data)*2, density=True, label='Data')
    
    result = minimize(negative_log_likelihood, x0=np.ones(1), args=(data,), method='Powell')
    
    print(result)
    
    x_plot = np.arange(0, max(data))
    
    fit = stats.poisson.pmf(x_plot, result.x)
    plt.plot(x_plot, fit, label=r'Poisson fit, $\mu$='+str(result.x[0]), color = 'black', linestyle = '--')
    plt.legend()
    plt.title('Histogram: Poisson Fit')
    plt.show()
    
    return fit

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def negative_log_likelihood(params, data):
    ''' better alternative using scipy '''
    return -stats.poisson.logpmf(data, params[0]).sum()
