#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
import scipy.odr as odr
from scipy import stats

import tools as tools


def odrfit(function, x, y,initials, xerr = None, yerr = None, param_mask = None,plot=False):

    model = odr.Model(function)
    inputdata = odr.RealData(x, y, sx = xerr, sy = yerr)

    odr_setup = odr.ODR(inputdata, model, beta0 = initials, ifixb = param_mask, sstol = 10e-3)
    odr_setup.set_job(fit_type=2, deriv=1)

    print('\nRunning fit!')
    output = odr_setup.run()
    print('\nFit done!')

    print('\nFitted parameters = ', output.beta)
    print('\nError of parameters =', output.sd_beta)

    if plot is True:
        plt.plot(x, function(output.beta, x))
        plt.plot(x, y)
        plt.show()

    return output.beta, output.sd_beta

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
    
    popt, pcov = odrfit(odr_lorentzian_lineshape, xdata, ydata, initials = pguess, param_mask = param_mask)
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


def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def lorentzian_lineshape(x, a, x0, gam, offset):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2) + offset

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
