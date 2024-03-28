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

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
DATA BUFFER FOR STREAMING MODE
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
Size of capture
'''
sizeOfOneBuffer = 500
numBuffersToCapture = 1
totalSamples = sizeOfOneBuffer * numBuffersToCapture
'''
Create buffer
'''
bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)

memory_segment = 0
'''
Set data buffer location for data collection from channel A
'''
handle = chandle
source = ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'] #0
bufferMax = bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
bufferMin = None
bufferLth = sizeOfOneBuffer
segmentIndex = memory_segment
mode = ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'] #0
status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(handle, source, bufferMax, bufferMin, bufferLth, segmentIndex, mode)
assert_pico_ok(status["setDataBuffersA"])

'''
Begin streaming mode
'''
handle = chandle
sampleInterval = ctypes.byref(ctypes.c_int32(250))
sampleIntervalTimeUnits = ps.PS5000A_TIME_UNITS['PS5000A_US']
# No trigger
maxPreTriggerSamples = 0
maxPostTriggerSamples = 0
autoStop = 1 #On=1, Off=0
# No downsampling
downSampleRatio = 1
downSampleRatioMode = ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE']
overviewBufferSize = sizeOfOneBuffer
status["runStreaming"] = ps.ps5000aRunStreaming(handle, sampleInterval, sampleIntervalTimeUnits, maxPreTriggerSamples, maxPostTriggerSamples, autoStop, downSampleRatio, downSampleRatioMode, overviewBufferSize)
assert_pico_ok(status["runStreaming"])

actualSampleInterval = sampleInterval.value
actualSampleIntervalNs = actualSampleInterval * 1000

print("Capturing at sample interval %s ns" % actualSampleIntervalNs)

'''
Set up large buffer, not registered with the driver, to keep our complete capture in.
'''
bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
nextSample = 0
autoStopOuter = False
wasCalledBack = False


def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
    global nextSample, autoStopOuter, wasCalledBack
    wasCalledBack = True
    destEnd = nextSample + noOfSamples
    sourceEnd = startIndex + noOfSamples
    bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
    nextSample += noOfSamples
    if autoStop:
        autoStopOuter = True

'''
Convert the python function into a C function pointer.
'''
cFuncPtr = ps.StreamingReadyType(streaming_callback)
'''
Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
'''
while nextSample < totalSamples and not autoStopOuter:
    wasCalledBack = False
    status["getStreamingLastestValues"] = ps.ps5000aGetStreamingLatestValues(chandle, cFuncPtr, None)
    if not wasCalledBack:
        # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
        # again.
        time.sleep(0.01)

print("Done grabbing values.")

'''
find maximum ADC count value
'''
handle = chandle
maxADC = ctypes.c_int16()
pointer_to_value = ctypes.byref(maxADC)
status["maximumValue"] = ps.ps5000aMaximumValue( handle, pointer_to_value)
assert_pico_ok(status["maximumValue"])

'''
convert ADC counts data to mV
'''
adc2mVChAMax = adc2mV(bufferCompleteA, chARange, maxADC)

'''
Create time data
'''
time = np.linspace(0, (totalSamples - 1) * actualSampleIntervalNs, totalSamples)

'''
Save data from channel A
'''
np.save("output/channelA_data_streaming.npy", np.array([time, adc2mVChAMax[:]]))
'''
Save buffer data
'''

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


