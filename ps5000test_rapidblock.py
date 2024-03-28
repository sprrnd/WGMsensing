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
import csv
import pickle




'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
DEFINE DATA COLLECTION PARAMETERS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
date = '2024-02-16'
n_frames = 10000
sample_no = 62500 #500us domain for 8ns timebase
timebase = 3 #minimum=8ns=3
sampling_interval = (timebase-2) / 125000000
frequency_range = (1)/(sampling_interval*2)

print('Collecting {} frames'.format(n_frames))
print('{}ns timebase'.format(sampling_interval))
print('{}MHz frequency range'.format(frequency_range*10**-6))
print('External Trigger: 2V, FALLING EDGE')

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
RAPID BLOCK MODE
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
# handle = chandle
# enabled = 1
# source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
# threshold = int(mV2adc(500,chARange, maxADC))
# direction = 2 #PS5000A_RISING = 2
# delay = 0 #s
# auto_Trigger = 1000 #ms
# status["trigger"] = ps.ps5000aSetSimpleTrigger(handle, enabled, source, threshold, direction, delay, auto_Trigger)
# assert_pico_ok(status["trigger"])

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

'''
Set number of pre and post trigger samples to be collected
'''
preTriggerSamples = int(sample_no/2)
postTriggerSamples = int(sample_no/2)
maxSamples = preTriggerSamples + postTriggerSamples


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
TIMEBASES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
Get timebase information
    = sampling interval of oscilloscope
'''
handle = chandle
timebase = timebase #minimum=8ns=3
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

# Creates a overlow location for data
overflow = ctypes.c_int16()
# Creates converted types maxsamples
cmaxSamples = ctypes.c_int32(maxSamples)
'''
Set number of memory segments = or > number of captures
'''
handle = chandle
nSegments = n_frames
nMaxSamples = ctypes.byref(cmaxSamples)
status["MemorySegments"] = ps.ps5000aMemorySegments(handle, nSegments, nMaxSamples)
assert_pico_ok(status["MemorySegments"])
'''
Set number of waveforms to capture before each run
'''
handle = chandle
nCaptures = n_frames
status["SetNoOfCaptures"] = ps.ps5000aSetNoOfCaptures(handle, nCaptures)
assert_pico_ok(status["SetNoOfCaptures"])


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
START CAPTURE DATA
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''
# Run block capture
# '''
# handle = chandle
# noOfPreTriggerSamples = preTriggerSamples
# noOfPostTriggerSamples = postTriggerSamples
# timebase = 4 #= 4 ns (see Programmer's guide for mre information on timebases)
# timeIndisposedMs = None #(not needed in the example)
# segmentIndex = 0
# lpReady = None #(using ps5000aIsReady rather than ps5000aBlockReady)
# pParameter = None
# status["runBlock"] = ps.ps5000aRunBlock(handle, noOfPreTriggerSamples, noOfPostTriggerSamples, timebase, timeIndisposedMs, segmentIndex, lpReady, pParameter)
# assert_pico_ok(status["runBlock"])

# '''
# Create buffers 0 ready for assigning pointers for data collection
# '''
# bufferAMax0 = (ctypes.c_int16 * maxSamples)()
# bufferAMin0 = (ctypes.c_int16 * maxSamples)()
# '''
# Set data buffer 0 location for data collection from channel A
# '''
# handle = chandle
# source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
# bufferMax = ctypes.byref(bufferAMax0)
# bufferMin = ctypes.byref(bufferAMin0)
# bufferLth = maxSamples
# segmentIndex = 0
# mode = 0
# status["SetDataBuffers"] = ps.ps5000aSetDataBuffers(handle, source, bufferMax, bufferMin, bufferLth, segmentIndex, mode)
# assert_pico_ok(status["SetDataBuffers"])

# '''
# Create set of buffers 1 ready for assigning pointers for data collection
# '''
# bufferAMax1 = (ctypes.c_int16 * maxSamples)()
# bufferAMin1 = (ctypes.c_int16 * maxSamples)()
# '''
# Set data buffer location 1 for data collection from channel A
# '''
# handle = chandle
# source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
# bufferMax = ctypes.byref(bufferAMax1)
# bufferMin = ctypes.byref(bufferAMin1)
# bufferLth = maxSamples
# segmentIndex = 1
# mode = 0
# status["SetDataBuffers"] = ps.ps5000aSetDataBuffers(handle, source, bufferMax, bufferMin, bufferLth, segmentIndex, mode)
# assert_pico_ok(status["SetDataBuffers"])

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
END CAPTURE DATA
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
LOOP CAPTURE DATA
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
Run block capture
'''
handle = chandle
noOfPreTriggerSamples = preTriggerSamples
noOfPostTriggerSamples = postTriggerSamples
timebase = timebase
timeIndisposedMs = None #(not needed in the example)
segmentIndex = 0
lpReady = None #(using ps5000aIsReady rather than ps5000aBlockReady)
pParameter = None
status["runBlock"] = ps.ps5000aRunBlock(handle, noOfPreTriggerSamples, noOfPostTriggerSamples, timebase, timeIndisposedMs, segmentIndex, lpReady, pParameter)
assert_pico_ok(status["runBlock"])


adc2mVChAMax_list = []
for n in range(n_frames):
    if n==0:
        print('DATA COLLECTION STARTED')
    
    if type(n/100) is int:
        print(n, ' samples collected')
    '''
    Create buffers ready for assigning pointers for data collection
    '''
    bufferAMaxn = (ctypes.c_int16 * maxSamples)()
    bufferAMinn = (ctypes.c_int16 * maxSamples)()
    '''
    Set data buffer 0 location for data collection from channel A
    '''
    handle = chandle
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    bufferMax = ctypes.byref(bufferAMaxn)
    bufferMin = ctypes.byref(bufferAMinn)
    bufferLth = maxSamples
    segmentIndex = n
    mode = ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'] # = 0
    status["setDataBuffers"] = ps.ps5000aSetDataBuffers(handle, source, bufferMax, bufferMin, bufferLth, segmentIndex, mode)
    assert_pico_ok(status["setDataBuffers"])
    
    
    '''
    create overflow location
    '''
    overflow = (ctypes.c_int16 * n_frames)()
    '''
    create converted type maxSamples
    '''
    cmaxSamples = ctypes.c_int32(maxSamples)
    '''
    Check data collection to finish the capture
    '''
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))

    
    '''
    Retrieve data from scope to buffers assigned above
    '''
    handle = chandle
    startIndex = 0
    noOfSamples = ctypes.byref(cmaxSamples)
    downSampleRatio = 0
    downSampleRatioMode = 0
    segmentIndex = n
    overflow = ctypes.byref(overflow)

    status["GetValues"] = ps.ps5000aGetValues(handle, startIndex, noOfSamples, downSampleRatio, downSampleRatioMode, segmentIndex, overflow)
    assert_pico_ok(status["GetValues"])
    
    adc2mVChAMaxn =  adc2mV(bufferAMaxn, chARange, maxADC) #mV
    raw_signal = [i/1000 for i in adc2mVChAMaxn] #V
    
    adc2mVChAMax_list.append(raw_signal)
    

    with open("output/"+date+"/data/raw_data_cumul_{}_nframes{}.txt".format(date, n_frames), 'ab') as file:
        # writer = csv.writer(file)
        # writer.writerow(raw_signal)
        pickle.dump(raw_signal, file)
        
np.save("output/"+date+"/data/raw_data_{}_nframes{}.npy".format(date, n_frames), np.array(adc2mVChAMax_list))

timebase_s = (timebase-2)/125000000
collection_time = timebase_s * maxSamples * n_frames
print(n_frames, ' frames collected over ', collection_time,' s')

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
EXIT LOOP CAPTURE DATA
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# '''
# create overflow location
# '''
# overflow = (ctypes.c_int16 * n_frames)()
# '''
# create converted type maxSamples
# '''
# cmaxSamples = ctypes.c_int32(maxSamples)
# '''
# Check data collection to finish the capture
# '''
# ready = ctypes.c_int16(0)
# check = ctypes.c_int16(0)
# while ready.value == check.value:
#     status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))

'''
Retrieve data from scope to buffers assigned above
'''
# handle = chandle
# noOfSamples = ctypes.byref(cmaxSamples)
# fromSegmentIndex = 0
# toSegmentIndex = n_frames - 1
# downSampleRatio = 0
# downSampleRatioMode = 0
# overflow = ctypes.byref(overflow)

# status["GetValuesBulk"] = ps.ps5000aGetValuesBulk(handle, noOfSamples, fromSegmentIndex, toSegmentIndex, downSampleRatio, downSampleRatioMode, overflow)
# assert_pico_ok(status["GetValuesBulk"])

'''
Retrieve trigger time stamps
'''
handle = chandle
times = (ctypes.c_int64*n_frames)()
Time = ctypes.byref(times)
timeUnits = ctypes.byref(ctypes.c_char())
fromSegmentIndex = 0
toSegmentIndex = n_frames - 1
status["GetValuesTriggerTimeOffsetBulk"] = ps.ps5000aGetValuesTriggerTimeOffsetBulk64(handle, Time, timeUnits, fromSegmentIndex, toSegmentIndex)
assert_pico_ok(status["GetValuesTriggerTimeOffsetBulk"])

'''
Get and print TriggerInfo for memory segments
Create array of ps.PS5000A_TRIGGER_INFO for each memory segment
'''
N_TriggerInfo = (ps.PS5000A_TRIGGER_INFO*n_frames) ()

handle = chandle
triggerInfo = ctypes.byref(N_TriggerInfo)
fromSegmentIndex = 0
toSegmentIndex = n_frames - 1
status["GetTriggerInfoBulk"] = ps.ps5000aGetTriggerInfoBulk(handle, triggerInfo, fromSegmentIndex, toSegmentIndex)
assert_pico_ok(status["GetTriggerInfoBulk"])

print("Printing triggerInfo blocks")
for i in N_TriggerInfo:
    print("PICO_STATUS is ", i.status)
    print("segmentIndex is ", i.segmentIndex)
    print("triggerTime is ", i.triggerTime)
    print("timeUnits is ", i.timeUnits)
    print("timeStampCounter is ", i.timeStampCounter)

'''
convert ADC counts data to mV
'''
# adc2mVChAMax0 =  adc2mV(bufferAMax0, chARange, maxADC)
# adc2mVChAMax1 =  adc2mV(bufferAMax1, chARange, maxADC)

'''
Create time data
'''
time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)
time = np.linspace(-(cmaxSamples.value - 1) * timeIntervalns.value  * 0.5 * 10**-3, (cmaxSamples.value - 1) * timeIntervalns.value * 0.5 * 10**-3, cmaxSamples.value)

'''
Save data from channel A
'''
np.save("output/channelA_time.npy", time)
# np.save("output/channelA_data0.npy", np.array([time, adc2mVChAMax0[:]]))
# np.save("output/channelA_data1.npy", np.array([time, adc2mVChAMax0[:]]))

'''
Plot data from channel A
'''
# plt.plot(time, adc2mVChAMax0[:])
# plt.plot(time, adc2mVChAMax1[:])

# for f, frame in enumerate(adc2mVChAMax_list):
#     plt.plot(time, frame[:])
#     plt.xlabel('Time (us)')
#     plt.ylabel('Voltage (V)')
#     # plt.savefig('output/channelA_data{}.png'.format(str(f)))
#     plt.show()

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



