#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:53:21 2024

@author: sabrinaperrenoud
"""

import os
import sys
import pandas as pd

print('Test working')


def test():
    print('Test 2 working')


def crunch_data(directory):
    print('Here')
    
    foldernames = os.listdir(directory)
    
    foldernames = [i for i in foldernames if 'combined_data' not in i]
    
    print('Available folders: ',foldernames)
    
    for folder in foldernames:
        target_dir = directory + folder
        
        filenames = os.listdir(target_dir+'/')
        
        df = pd.read_csv(target_dir+ '/' + filenames[0])
        df.rename(columns={"Channel A": "{}".format(str(0))}, inplace=True)
        
        for i in range(len(filenames)):
            if i ==0:
                pass
            else:
                dfi = pd.read_csv(target_dir+'/'+filenames[i])
                dfi.rename(columns={"Channel A": "{}".format(str(i))}, inplace=True)
                df = df.merge(dfi, on="Frequency", how='inner')
        saveas='combined_data_{a}'.format(a=str(folder))
        df.to_csv(directory+str(saveas)+'.csv', index = False)
    return df
    
test()
crunch_data('folder/')