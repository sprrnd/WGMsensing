#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from natsort import natsorted
import numpy as np
import pandas as pd
import sys
import scipy
from scipy.io import loadmat


def mac_natsorted(list):

    output = natsorted(list)
    if '.DS_Store' in output: output.remove('.DS_Store')
    return output



def load_mat_file(directory, mac = False):
    '''
    Function: Loads all .mat files in given folder
    Input:
        directory = path to files
        mac = True/False specific for mac user
    Output:
        data_dictionary = dictionary of data for all .mat files
    '''

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
    '''
    Function: Loads all .csv files in given folder
    Input:
        directory = path to files
        mac = True/False specific for mac user
    Output:
        data_dictionary = dictionary of data for all .csv files
    '''
    if mac is True:
        filenames = mac_natsorted(os.listdir(directory))
    else:
        filenames = os.listdir(directory)
    filenames = [i for i in filenames if '.csv' in i and '._' not in i]
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
        
        data_dictionary[f] = [frequency, signal_list]

    return data_dictionary

def load_txt_file(directory, skiprows, mac = False):
    '''
    Function: Loads all .txt files in given folder
    Input:
        directory = path to files
        mac = True/False specific for mac user
    Output:
        data_dictionary = dictionary of data for all .txt files
    '''
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
