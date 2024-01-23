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
from itertools import zip_longest

import tools as tools



def mac_natsorted(list):

    output = natsorted(list)
    if '.DS_Store' in output: output.remove('.DS_Store')
    return output


@numba.jit
def crunch_data(directory, crop=None):
    filenames = mac_natsorted(os.listdir(directory))[:int(crop)]
    print('Files: ',filenames)
    for filename in filenames:
        if 'csv' in filename:
            pass
        else:
            datasetnames = mac_natsorted(os.listdir(directory+filename))
            # print('Datasets: ',datasetnames)
            df = pd.read_csv(directory+filename+'/'+datasetnames[0])
            df.rename(columns={"Channel A": "{}".format(str(0))}, inplace=True)
            for i in range(len(datasetnames[:crop])):
                # print(str(int(100*i/len(datasetnames)))+'%')
                if i ==0:
                    pass
                else:
                    dfi = pd.read_csv(directory+filename+'/'+datasetnames[i])
                    dfi.rename(columns={"Channel A": "{}".format(str(i))}, inplace=True)
                    df = df.merge(dfi, on="Frequency", how='inner')
            df.to_csv('data/output/'+'{}_combined.csv'.format(str(filename)), index = False)
            print('Combined Dataset: ',df)
    return df

@numba.jit
def save_crunch_data(directory, folder, crop=[0,-1]):

    link = directory+folder
    datasetnames = mac_natsorted(os.listdir(link))[crop[0]:crop[1]]
    # print(datasetnames)
    
    df = pd.read_csv(link+'/'+datasetnames[0])
    df.rename(columns={"Channel A": "{}".format(str(0))}, inplace=True)
    
    for i in range(len(datasetnames)):
        if i ==0:
            pass
        else:
            dfi = pd.read_csv(link+'/'+datasetnames[i])
            dfi.rename(columns={"Channel A": "{}".format(str(i))}, inplace=True)
            df = df.merge(dfi, on="Frequency", how='inner')
    saveas='saved_data_{a}_{b}'.format(a=str(folder), b=str(crop))
    df.to_csv(directory+'saved_data/'+str(saveas)+'.csv', index = False)
    # print('Combined Dataset: ',df)
    return df

# @njit
# def classify_data(target_dir, filenames):
    
#     # frequency = []
#     signal = []
#     info = []
    
#     frequency = np.loadtxt(target_dir+'/'+filenames[0], delimiter =',', usecols=0,skiprows=2)
    
    
#     for filename in filenames:
#         print(filename)
#         # data0 = np.loadtxt(target_dir+'/'+filename, delimiter =',', usecols=0,skiprows=2)
#         data1 = np.loadtxt(target_dir+'/'+filename, delimiter =',', usecols=1,skiprows=2)
        
#         crunched = pd.read_csv(directory+"saved_data/output/2024-01-11_crunched_data_100nMGaba_H2O_1_1000.csv")

#         name = filename.strip('.csv')
#         string_index = name.find('bg')
#         info.append(filename[string_index :string_index + 4])

        
        
#         # frequency.append(data0)
#         signal.append(data1)
        
#         info.append(name)

#     return frequency, signal, info

def classify_data(target_dir):
    
    filenames = mac_natsorted(os.listdir(target_dir))
    
    data = []
    info = []
    
    for filename in filenames:
        print(filename)
        
        crunched = pd.read_csv(target_dir+filename)

        name = filename.strip('.csv')
        string_index = name.find('bg')
        info.append(filename[string_index :string_index + 4])


        data.append(crunched)
        
        info.append(name)

    return data


# @njit
def classify_file(target_dir, foldernames, crop):
    
    bg = []
    data = []
    
    for f in foldernames:
        
        if 'saved_data' in f:
            pass
        
        elif 'bg' in f:
            # bg.append(classify_data(target_dir+f+'/', filenames))
            save = save_crunch_data(target_dir, f)
            load = classify_data(target_dir+'saved_data/')
            bg.append(load)       
        else:
            save = save_crunch_data(target_dir, f, crop)
            load = classify_data(target_dir+'saved_data/')
            data.append(load)           
    
    return bg, data


def save_file(directory, foldernames, crop, bg=True):
    
    
    for f in foldernames:
        
        if 'saved_data' in f:
            pass
        
        elif 'bg' in f:
            if bg is True:
                save = save_crunch_data(directory, f)
            else:
                pass
        else:
            save = save_crunch_data(directory, f, crop)        

def load_file_simple(directory):
    
    filenames = mac_natsorted(os.listdir(directory+"saved_data/"))
    print(filenames)
    
    bg = []
    data = []
    
    for f in filenames:
        
        crunched = pd.read_csv(directory+"saved_data/"+f)
        frequency = crunched.iloc[:,[0]]
        frequency = frequency['Frequency'].tolist()[1:]
        frequency = np.array([float(f) for f in frequency])

        signals = crunched.iloc[:,1:]
        
        frame_no = len(crunched.loc[0])
        print(frame_no,' frames in dataset ', f)
        
        signal_list = []
        for i in range(frame_no-1):
            signal = signals[str(i)].tolist()[1:]
            signal_array = np.array([float(s) for s in signal if s!='-∞'])
            signal_list.append(signal_array)
        data.append([frequency, signal_list])

    return bg, data

def load_file(directory, crop, corr=False):
    
    filenames = mac_natsorted(os.listdir(directory+"saved_data/"))
    print(filenames)
    
    bg = []
    data = []
    gaba = []
    water = []
    
    for f in filenames:
        
        if 'bg' in f:
            load = pd.read_csv(directory+"saved_data/"+f)
            
            frequency = load.iloc[:,[0]]
            frequency = frequency['Frequency'].tolist()[1:]
            frequency = np.array([float(f) for f in frequency])
            
            bg_signals = load.iloc[:,1:]

            bgsignal_list = []
            for i in range(crop[1]-crop[0]):
                bgsignal = bg_signals[str(i)].tolist()[1:]
                bgsignal_array = np.array([float(s) for s in bgsignal])
                bgsignal_list.append(bgsignal_array)
            
            bg.append(frequency)
            bg.append(bgsignal_list)  
    
        
        elif str(crop) in f:
            crunched = pd.read_csv(directory+"saved_data/"+f)
            frequency = crunched.iloc[:,[0]]
            frequency = frequency['Frequency'].tolist()[1:]
            frequency = np.array([float(f) for f in frequency])

            signals = crunched.iloc[:,1:]

            signal_list = []
            for i in range(crop[1]-crop[0]):
                signal = signals[str(i)].tolist()[1:]
                signal_array = np.array([float(s) for s in signal])
                signal_list.append(signal_array)
            data.append([frequency, signal_list])
    # data_corr=[]
    # if corr is True:
    #     bg_avg = background_avg(bg)
    #     for dataset in data:
    #         data_corr_list=[]
    #         for i, arr in enumerate(dataset[1]):
    #             data_corr_list.append(np.array(arr[i]) - np.array(bg_avg))
    #         data_corr.append([dataset[0], data_corr_list])
    
    data_corr = []
    if corr is True:
        bg_avg = tools.background_avg(bg)
        for dataset in data:
            data_corr_list=[]
            for i, arr in enumerate(dataset[1]):
                data_corr_list.append(np.array(arr) - np.array(bg_avg))
            data_corr.append([frequency, data_corr_list])
                
    return bg, data, data_corr

def combine_files(directory):
    filenames = mac_natsorted(os.listdir(directory+'saved_data/'))
    print(filenames)
    
    gaba=[]
    gaba_corr=[]
    crop_list = []
    for i,f in enumerate(filenames):
        if 'Gaba' in f:
            crop = f.split('.')[0].split('_')[-1]
            a = crop.strip('[]')
            b, c = a.split(', ')
            crop = [int(b), int(c)]
            crop_list.append([int(b), int(c)])
            
            bg, data, data_corr = load_file(directory, crop, corr=True)
            
            gaba.append(data[0][0])
            gaba.append([])
            gaba[1] += data[0][1]
            gaba_corr.append(data_corr[0][0])
            gaba_corr.append([])
            gaba_corr[1] += data_corr[0][1]
    print(crop_list)
    return crop_list, gaba, gaba_corr

def combine_fit_parameters(target_dir):
    frame_list = []
    filenames = mac_natsorted(os.listdir(target_dir))
    
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
    np.save(target_dir+"fit_parameters.npy", [xframes, amplitudes, frequencies, widths, fits]) 
