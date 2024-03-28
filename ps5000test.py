#
# Copyright (C) 2018-2022 Pico Technology Ltd. See LICENSE file for terms.
#
'''
PS5000A BLOCK MODE EXAMPLE
This example opens a 5000a driver device, sets up two channels and a trigger then collects a block of data.
This data is then plotted as mV against time in ns.
'''

import picosdk

import ctypes
import numpy as np
from picosdk.ps5000a import ps5000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc
import time
import ps5000_functions as func

chandle = ctypes.c_int16()
status = {}

resolution =ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)

try:
    assert_pico_ok(status["openunit"])
except: # PicoNotOkError:

    powerStatus = status["openunit"]

    if powerStatus == 286:
        status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
    elif powerStatus == 282:
        status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
    else:
        raise

    assert_pico_ok(status["changePowerSource"])

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SET UP CHANNELS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
Set up Channel A
'''
handle = chandle
channel = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
enabled = 1
disabled = 0
coupling_type = ps.PS5000A_COUPLING["PS5000A_DC"]
chARange = ps.PS5000A_RANGE["PS5000A_2V"]
analogue_offset = 0 #V
status["setChA"] = ps.ps5000aSetChannel(handle, channel, enabled, coupling_type, chARange, analogue_offset)
assert_pico_ok(status["setChA"])

'''
find maximum ADC count value
'''
handle = chandle
maxADC = ctypes.c_int16()
pointer_to_value = ctypes.byref(maxADC)
status["maximumValue"] = ps.ps5000aMaximumValue( handle, pointer_to_value)
assert_pico_ok(status["maximumValue"])

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SINGLE TRIGGER
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
handle = chandle
enabled = 1
source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
threshold = int(mV2adc(500,chARange, maxADC))
direction = 2 #PS5000A_RISING = 2
delay = 0 #s
auto_Trigger = 1000 #ms
status["trigger"] = ps.ps5000aSetSimpleTrigger(handle, enabled, source, threshold, direction, delay, auto_Trigger)
assert_pico_ok(status["trigger"])

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SINGLE EXTERNAL TRIGGER
    source = EXTERNAL
    threshold = 2V
    direction = falling
    pre-trigger = 50%
    mode = auto
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
handle = chandle
enabled = 1
source = ps.PS5000A_CHANNEL["PS5000A_EXTERNAL"]
threshold = int(mV2adc(2000,chARange, maxADC)) #mV2adc = millivolts to adc (2V=2000mV)
direction = 3 #PS5000A_RISING = 2, FALLING = 3
delay = 0 #s
auto_Trigger = 0 #ms
status["trigger"] = ps.ps5000aSetSimpleTrigger(handle, enabled, source, threshold, direction, delay, auto_Trigger)
assert_pico_ok(status["trigger"])


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
ADVANCED TRIGGER
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# '''
# Set Trigger Properties
# '''
# thresholdUpper = mV2adc(500, chARange, maxADC) #adcTriggerLevel
# thresholdUpperHysteresis = 10
# thresholdLower = 0
# thresholdLowerHysteresis = 10
# channel = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
# triggerProperties = ps.PS5000A_TRIGGER_CHANNEL_PROPERTIES_V2(thresholdUpper, thresholdUpperHysteresis, thresholdUpperHysteresis, thresholdLower, thresholdLowerHysteresis, channel)													

# handle = chandle
# channelProperties = ctypes.byref(triggerProperties)
# nChannelProperties = 1
# auxOutputEnable = 0
# status["setTriggerChannelPropertiesV2"] = ps.ps5000aSetTriggerChannelPropertiesV2(chandle, ctypes.byref(triggerProperties), nChannelProperties, auxOutputEnable)
# '''
# Set Trigger Conditions
# '''
# source = ps.PS5000A_CHANNEL["PS5000A_EXTERNAL"]
# condition = ps.PS5000A_TRIGGER_STATE["PS5000A_CONDITION_TRUE"]
# triggerConditions = ps.PS5000A_CONDITION(source, condition)

# clear = 1
# add = 2
# handle = chandle
# conditions = ctypes.byref(triggerConditions)
# nConditions = 1
# info = (clear + add)													   
# status["setTriggerChannelConditionsV2"] = ps.ps5000aSetTriggerChannelConditionsV2(handle, conditions, nConditions, info)
# '''
# Set Trigger Directions
# '''
# source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
# direction = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]
# mode = ps.PS5000A_THRESHOLD_MODE["PS5000A_LEVEL"]
# triggerDirections = ps.PS5000A_DIRECTION(source, direction , mode)
# status["setTriggerChannelDirections"] = ps.ps5000aSetTriggerChannelDirectionsV2(chandle, ctypes.byref(triggerDirections), 1)

'''
Set number of pre and post trigger samples to be collected
'''
sample_no = 62500
preTriggerSamples = sample_no/2
postTriggerSamples = sample_no/2
maxSamples = preTriggerSamples + postTriggerSamples

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
TIMEBASES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
Get minimum timebase information
'''
handle = chandle
enabledChannelOrPortFlags = 1 #channelA
timebase = 1
timeInterval = 1
resolution = ps.PS5000a_DEVICE_RESOLUTION['PS5000A_DR_12BIT']
status["getMinimumTimebaseStateless"] = ps.ps5000aGetMinimumTimebaseStateless(handle, enabledChannelOrPortFlags, timebase, timeInterval, resolution)
assert_pico_ok(status["getMinimumTimebaseStateless"])
'''
Set timebase
    = sampling interval of oscilloscope
'''
handle = chandle
timebase = 0
noSamples = maxSamples
timeIntervalns = ctypes.c_float()
returnedMaxSamples = ctypes.c_int32()
pointer_to_timeIntervalNanoseconds = ctypes.byref(timeIntervalns)
pointer_to_maxSamples = ctypes.byref(returnedMaxSamples)
segment_index = 0
status["getTimebase2"] = ps.ps5000aGetTimebase2(handle, timebase, noSamples, pointer_to_timeIntervalNanoseconds, pointer_to_maxSamples, segment_index)
assert_pico_ok(status["getTimebase2"])

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SETUP DATA TRANSFER
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
handle = chandle
startIndex = 0
noOfSamples = maxSamples
downSampleRatio = 0
downSampleRatioMode = ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE']
segmentIndex = 0
overflow= ctypes.byref(ctypes.c_int16())
status["GetValuesOverlapped"] = ps.ps5000aGetValuesOverlapped(handle, startIndex, noOfSamples, downSampleRatio, downSampleRatioMode, segmentIndex, overflow)
assert_pico_ok(status["GetValuesOverlapped"])
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CAPTURE DATA
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
Run block capture
'''
handle = chandle
noOfPreTriggerSamples = preTriggerSamples
noOfPostTriggerSamples = postTriggerSamples
timebase = 0 #= 1 ns (see Programmer's guide for mre information on timebases)
timeIndisposedMs = None #(not needed in the example)
segmentIndex = 0
lpReady = None #(using ps5000aIsReady rather than ps5000aBlockReady)
pParameter = None
status["runBlock"] = ps.ps5000aRunBlock(handle, noOfPreTriggerSamples, noOfPostTriggerSamples, timebase, timeIndisposedMs, segmentIndex, lpReady, pParameter)
assert_pico_ok(status["runBlock"])

'''
Check for data collection to finish using ps5000aIsReady
'''
ready = ctypes.c_int16(0)
check = ctypes.c_int16(0)
while ready.value == check.value:
    status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))

'''
Create buffers ready for assigning pointers for data collection
'''
bufferAMax = (ctypes.c_int16 * maxSamples)()
bufferAMin = (ctypes.c_int16 * maxSamples)() # used for downsampling which isn't in the scope of this example

'''
Set data buffer location for data collection from channel A
'''
handle = chandle
source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
bufferMax = ctypes.byref(bufferAMax)
bufferMin = ctypes.byref(bufferAMin)
bufferLth = maxSamples
segmentIndex = 0
mode = ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'] # = 0
status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(handle, source, bufferMax, bufferMin, bufferLth, segmentIndex, mode)
assert_pico_ok(status["setDataBuffersA"])

'''
create overflow loaction
'''
overflow = ctypes.c_int16()

'''
create converted type maxSamples
'''
cmaxSamples = ctypes.c_int32(maxSamples)

'''
Retried data from scope to buffers assigned above
'''
handle = chandle
startIndex = 0
noOfSamples = ctypes.byref(cmaxSamples)
downSampleRatio = 0
downSampleRatioMode = ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE']
segmentIndex = 0
overflow= ctypes.byref(overflow)
status["getValues"] = ps.ps5000aGetValues(handle, startIndex, noOfSamples, downSampleRatio, downSampleRatioMode, segmentIndex, overflow)
assert_pico_ok(status["getValues"])

'''
convert ADC counts data to mV
'''
adc2mVChAMax =  adc2mV(bufferAMax, chARange, maxADC)

'''
Create time data
'''
time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)

'''
Save data from channel A
'''
np.save("output/channelA_data.npy", np.array([time, adc2mVChAMax[:]]))

'''
Plot data from channel A
'''
plt.plot(time, adc2mVChAMax[:])
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (mV)')
plt.savefig('output/channelA_data.png')
plt.show()



'''
FFT Spectrum
'''
import scipy.fftpack

# Number of sample points
N = (cmaxSamples.value - 1)
# Sample spacing
sampling_interval = (timebase-3) / 62500000
T = timeIntervalns.value
x = time
y = adc2mVChAMax[:]
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.savefig('output/channelA_fft.png')
plt.show()

np.save("output/channelA_fft.npy", yf)



'''
Stop the scope
'''
handle = chandle
status["stop"] = ps.ps5000aStop(handle)
assert_pico_ok(status["stop"])

'''
Close unit Disconnect the scope
'''
handle = chandle
status["close"]=ps.ps5000aCloseUnit(handle)
assert_pico_ok(status["close"])

'''
Display status returns
'''
print(status)


#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

def dBu2V(dbu):
    #dbu = 0.775 * V
    V = dbu/0.775
    return V

def V2dBu(V):
    dBu = 0.775 * V
    return dBu

def plot_fft(x, y):
    time = x
    N = (len(y) - 1)
    T = time[1]-time[0]
    y = y
    xf = scipy.fftpack.fftfreq(N, T)[:N//2]
    yf = scipy.fftpack.fft(y)
    
    # yf = V2dBu(yf)
    yf = 20*np.log10(yf)
    
    # xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fig, ax = plt.subplots()
    # ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    ax.plot(xf, yf[:N//2])
    # plt.savefig('output/channelA_fft.png')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Intensity (dB)')
    plt.show()

def get_fft(x, data):
    fft_signals=[]
    for s, signal in enumerate(data):
        y = signal
        N = (len(y) - 1)
        T = time[1]-time[0]
        xf = scipy.fftpack.fftfreq(N, T)[:N//2]
        yf = scipy.fftpack.fft(y)
        
        # yf = V2dBu(yf)
        yf = 20*np.log10(yf)
        yf = yf[:N//2]
        fft_signals.append(yf)    
    
        
    np.save('output/channelA_fft.npy', np.array([xf, fft_signals]), dtype=object)
    return fft_signals

time = np.load('output/channelA_time.npy', allow_pickle = True)


n_frames = 100
import pickle

dataopen2s=[]
with open("output/channelA_datan_open.txt", 'rb') as file:
    # reader = csv.reader(file)
    # dataopen2 = list(reader)
    for _ in range(n_frames):
        dataopen2 = pickle.load(file)
        dataopen2s.append(dataopen2)

data2 = np.array(dataopen2s)
data = np.load('output/channelA_datan.npy', allow_pickle = True)


plt.plot(time, data[0])
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.show()

plot_fft(time, data[0])

#%%

get_fft(time, data)
fft_datasets = np.load('output/channelA_fft.npy', allow_pickle = True)
