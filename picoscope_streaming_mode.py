#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:42:34 2024

@author: sabrinaperrenoud
"""

# picoscope /c *.psdata /f mat /b all

import ctypes
import numpy as np
from picosdk.ps6000 import ps6000 as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
import time



from picosdk.discover import find_all_units

scopes = find_all_units()

for scope in scopes:
    print(scope.info)
    scope.close()





handle = ctypes.c_int16()
status = {}

status['openunit'] = ps.ps6000aOpenUnit(ctypes.byref(handle), None)


channelA = 'PICO_CHANNEL_A'
coupling = 'PICO_DC'
channelRange = 'PICO_CONNECT_PROBE_OFF'
analogueOffset = 0
bandwidth = 'PICO_BW_FILL'

ps.ps6000aSetChannelOn(handle, channelA, coupling, channelRange, analogueOffset, bandwidth)

for x in range(1,4):
    channel = x
    ps.ps6000aSetChannelOff(handle, x)
    
    

# Set number of samples to be collected
nSamples = 100

enable = 1
source = 'PICO_CHANNEL_A'
threshold = 100
direction = 'RISING_OR_FALLING'
delay = 0
auto_trigger = 1000000

ps.ps6000aSetSimpleTrigger(handle, enable, source, threshold, direction, delay, auto_trigger)

#Buffer
maxBuffers = 10
bufferA = ((c_int16 * nSamples) * 10)()


channel = 'PICO_CHANNEL_A'
buffer = bufferA
nSamples = nSamples
dataType = 'PICO_INT16_T'
waveform = 0
downSampleMode = 'PICO_RATIO_MODE_RAW'
action = 'PICO_CLEAR_ALL'


ps.ps6000aSetDataBuffer(handle, channel, buffer, nSamples, dataType, waveform, downSampleMode, action)


# Run streaming
sampleInterval = c_double(1)
sampleIntervalTimeUnits = 'PICO_US'
maxPreTriggerSamples = 50
maxPostTriggerSamples = 50
autoStop = 0
downSampleRatio = 1

ps.ps6000aRunStreaming(handle, sampleInterval, sampleIntervalTimeUnits, maxPreTriggerSamples, maxPostTriggerSamples, autoStop, downSampleRatio, downSampleMode)


channelA = 'PICO_CHANNEL_A'
streamingDataInfo = structs.PICO_STREAMING_DATA_INFO(channelA, downSampleMode, dataType, 0, 0, 0, 0)
triggerInfo = structs.PICO_STREAMING_DATA_TRIGGER_INFO(0, 0, 0)


ps.ps6000aGetStreamingLatestValues(handle, streamingDataInfo, 1, triggerInfo)


# stop scope streaming
ps.ps6000aStop(handle)

# get total number of streamed data points
noOfStreamedSamples=ctypes.c_uint64()
ps.ps6000aNoOfStreamingValues(handle, noOfStreamedSamples)


print("streaming finished")
print("Number of samples collected during streaming")
print(noOfStreamedSamples.value)


# Close the scope
ps.ps6000aCloseUnit(handle)
