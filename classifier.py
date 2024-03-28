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
import scipy
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
import numba
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
import scipy.odr as odr
from itertools import zip_longest
import sys
from scipy.io import loadmat

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

    
def load_mat_file(directory, mac = False):
    if mac is True:
        filenames = mac_natsorted(os.listdir(directory))
    else:
        filenames = os.listdir(directory)
    
    filenames = [i for i in filenames if '.mat' in i and '._' not in i]

    print('Available folders: ',filenames)
    
    data_folder = {}
    for file in filenames:
        name = str(file.strip('.mat'))
        
        mat = scipy.io.loadmat(directory+file)
        
        print('Tstart = ',mat['Tstart'])
        print('Tinterval = ',mat['Tinterval'])
        print('Length = ',mat['Length'])
        
        start = mat['Tstart'][0][0]
        inter =  mat['Tinterval'][0][0]
        length = mat['Length'][0][0]+4
        
        print(start, inter, length)
        
        
        todelete = [k for k in mat.keys() if k.isdigit() == False]
        print(todelete)
        for td in todelete:
            mat.pop(td)

    
        frequency = np.arange(start, start+( inter*length), inter)

        mat['Frequency'] = frequency
            
    
        data_folder[name] = mat  
    
    print('Available data files in data_folder: ', data_folder.keys())
    return data_folder

def load_csv_file(directory, mac = False):
    
    if mac is True:
        filenames = mac_natsorted(os.listdir(directory))
    else:
        filenames = os.listdir(directory)
    print(filenames)
    
    data_dictionary = {}
    
    for f in filenames:
        
        crunched = pd.read_csv(directory+f, delimiter = ',', engine ='python', dtype = object)
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
        
        fname = f.strip('.csv')
        data_dictionary[fname] = [frequency, signal_list]

    return data_dictionary

def load_txt_file(directory, skiprows, mac = False):
    if mac is True:
        filenames = mac_natsorted(os.listdir(directory))
    else:
        filenames = os.listdir(directory)
    
    filenames = [i for i in filenames if '.txt' in i and '._' not in i]
    print('Available folders: ',filenames)
    
    data_folder = {}
    for file in filenames:
        name = str(file.strip('.txt'))
        
        datafile = np.loadtxt(directory+file, delimiter='\t', skiprows = skiprows)
    
        data_folder[name] = datafile 
    
    print('Available data files in data_folder: ', data_folder.keys())
    
    return data_folder

def correct_file(bg, data):
    data_corr = []
    bg_avg = tools.background_avg(bg)
    for i, arr in enumerate(data[1]):
        data_corr.append(np.array(arr) - np.array(bg_avg))
    data_corr.append([frequency, data_corr_list])
                
    return bg, data, data_corr


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




def process_dict(dictionary):
    
    filenames = dictionary.keys()
    print(filenames)
    
    data = {}
    
    bg_avg = 0
    
    bg_filenames = [b for b in filenames if 'blank' in b or 'bg' in b]
    print('Available background filenames: ', bg_filenames)
    
    if len(bg_filenames) == 1:
        print('Background available to process')
    
        background = []
        bg = dictionary[bg_filenames[0]]
        bgfrequency = bg['Frequency']
        cutoff = int(len(bgfrequency)/2)
        bgfrequency = np.array([float(f) for f in bgfrequency])[:cutoff]
        background.append(bgfrequency)
        bgsignal_list = []
        for i in range(len(bg) - 1):
            bgsignal = bg[str(i)]
            bgsignal_array = np.array([float(s) for s in bgsignal])[:cutoff]
            bgsignal_list.append(bgsignal_array)
        background.append(bgsignal_list)
        
        bg_avg = tools.background_avg(background)
        
        data['bg'] = background
    
    
    for f in filenames:
        if f not in bg_filenames:
            load = dictionary[f]
            print(load.keys())
            frequency = load['Frequency']
            cutoff = int(len(frequency)/2)
            frequency = np.array([float(f) for f in frequency])[:cutoff]
            
            crop = [0, len(load) - 1]
    
            signal_list = []
            signal_corr_list = []
            for i in range(crop[1]-crop[0]):
                # print(i,' out of ', crop[1])
                if 'lock' in f:
                    signal = load[str(i)].T
                else:
                    signal = load[str(i)]
                    
                signal_array = np.array([float(s) for s in signal])[:cutoff]
                signal_list.append(signal_array)
                
                signal_corr = signal_array  - np.array(bg_avg)
                signal_corr_list.append(signal_corr)
                    
    
    
            data[f] = [frequency, signal_list]
            data[f+'_corr'] = [frequency, signal_corr_list]
                
                
    return data