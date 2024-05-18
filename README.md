# WGMsensing
### Data analysis pipelines for analysing data from WGM microresonator experiments.

## Main Dependencies
- numpy
- scipy
- matplotlib
- pandas
- natsort
- numba

Pip install or conda install can be used to install all of these packages

## Running the main data analysis
master = Main data analysis branch

## Master branch organisation

- HelperFunctions:
  Python files containing general functions shared and used by various Pipelines
    - fitting.py
    - classifier.py
    - tools.py
  
  These need to be included alongside a particular Pipeline in order to load in the relevant functions

- BeatnotePipelines:
  Methods and functions for the active beatnote data analysis
    - BeatnoteWorkbook.ipynb
    - BeatnotePipeline.py
    - BeatnoteMethods.py

- ResonanceShiftPipelines:
  Methods and functions for the passive resonance shift data analysis
    - ResonanceWorkbook.ipynb
    - ResonancePipeline.py
    - ResonanceMethods.py
- OSAPipelines:
  Methods and functions for plotting and analysing OSA spectrum data
    - OSAWorkbook.ipynb
    - OSAPipeline.py
    - OSAMethods.py
 
## Pipeline folders organisation

In each of those last three Pipelines folders, you will find three files:

### _Workbook.ipynb: 
  - this is the main Jupyter notebook for analysing the data
  - parameters can be defined and changed directly in the notebook
    
### _Pipeline.py:
  - a python class which contains all the high-level data handling and analysis called on by the Jupyter workbook
  - calls on various functions defined in its _Methods.py counterpart
  - to be referred to for any details about functions, inputs and outputs (hopefully should not need to be tweaked)

### _Methods.py:
  - a python file which contains all the relevant functions called on by _Pipeline.py
  - most of the processing is done here so if there is an error somewhere, this is probably the first place to check

## File structure

The code is defined to follow the following input and ouput structure:

    Workbook.ipynb
    
    Pipeline.py 
    
    Methods.py
    
    fitting.py
    
    tools.py
    
    classified.py
    
    data (folder)
    
        2024-03-31 (folder)
        
            data.csv
            
            data.mat
            
            data.txt
            
    output (folder)
    
        2024-03-31 (folder)
        
            plot.png
            
            saved_parameters.npy
