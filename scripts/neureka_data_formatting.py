import os
import re
import sys
import time
import math
import argparse
import pyedflib
from collections import OrderedDict
import numpy as np
from scipy import signal

 
DELIM_BCLOSE = '}'
DELIM_BOPEN = '{'
DELIM_CLOSE = ']'
DELIM_COLON = ':'
DELIM_COMMA = ','
DELIM_COMMENT = '#'
DELIM_EQUAL = '='
DELIM_NEWLINE = '\n'
DELIM_NULL = ''
DELIM_QUOTE = '"'
DELIM_SPACE = ' '
DELIM_SUB = "--"

KEYWORD_VERSION = "version"
KEYWORD_VERSION_NUMBER = "1.0"
KEYWORD_HELP = "-help"

KEYWORD_CSEL = "channel_selection"
KEYWORD_MONTAGE = "montage"
KEYWORD_MMODE = "match_mode"

KEYWORD_NULL = "(null)"
KEYWORD_EXACT = "exact"
KEYWORD_PARTIAL = "partial"

# This specifies the names of the edf and parameters files

################################################################################
# FUNCTIONS
################################################################################

# This loads an edf file and reads its frame rate, its signal and its channels' labels
def nedc_load_edf(fname_a):
  fp = pyedflib.EdfReader(fname_a)
  num_chans = fp.signals_in_file
  labels_tmp = fp.getSignalLabels()
  labels = [str(lbl.replace('  ', ' ')) for lbl in labels_tmp]

  sig = []
  fsamp = []
  for i in range(num_chans):
      sig.append(fp.readSignal(i))
      fsamp.append(fp.getSampleFrequency(i))
  return (fsamp, sig, labels)


# Match a channel label to the correct channel
def nedc_get_pos(lbl_a, labels_a, mmode_a):
  indices = []
  if mmode_a == KEYWORD_EXACT:
      pos = labels_a.index(lbl_a)
      if pos >= int(0):
          indices.append(pos)
      else:
          indices.append(int(-1))
  else:
      indices = [i for i, elem in enumerate(labels_a) if lbl_a in elem]
  if len(indices) == 0:
      return int(-1)
  else:
      return indices[0]


# This selects only the channels that are specified in the parameters file
def nedc_select_channels(params_a, fsamp_a, sig_a, labels_a):
    fsamp_sel = []
    sig_sel = []
    labels_sel = []
    # extract the list of channels from the parameter block
    chan_list = params_a.get(KEYWORD_CSEL).split(DELIM_COMMA)
    # if the channel list contains null, simply copy the input to the output
    if KEYWORD_NULL in chan_list:
        return (fsamp_a, sig_a, labels_a)
    for lbl in chan_list:
        # look up the label in the original signal
        pos = nedc_get_pos(lbl, labels_a, params_a[KEYWORD_MMODE])
        # append the corresponding signal
        if pos >= int(0):
            fsamp_sel.append(fsamp_a[pos])
            sig_sel.append(sig_a[pos])
            labels_sel.append(labels_a[pos])
        else:
            print("Can't find label {}".format(lbl))
    return fsamp_sel, sig_sel, labels_sel


# This transforms the montage in the params file into machine readable data
def nedc_parse_montage(params_a):
     montage = []
     for str in params_a[KEYWORD_MONTAGE]:
          # split the line into subparts
          parts = str.split(DELIM_COMMA)          
          subparts = parts[1].split(DELIM_COLON)
          expparts = subparts[1].split(DELIM_SUB)

          # assemble it into a full list
          parts[1] = subparts[0]
          parts.append(expparts[0])
          if len(expparts) > 1:
               parts.append(expparts[1])
          else:
               parts.append(KEYWORD_NULL)
          montage.append(parts)
     return montage


# This applies the montage that is specified in the parameters file
def nedc_apply_montage(params_a, fsamp_a, sig_a, labels_a):
  fsamp_mont = []
  sig_mont = []
  labels_mont = []
  if KEYWORD_NULL in params_a[KEYWORD_MONTAGE.lower()]:
      return (fsamp_a, sig_a, labels_a)
  # convert the raw format of the montage into something easy to process
  montage = nedc_parse_montage(params_a)
  for i in range(len(montage)):
      # get the position of the first operand
      pos1 = nedc_get_pos(montage[i][2], labels_a, params_a[KEYWORD_MMODE])
      if montage[i][3] != KEYWORD_NULL:
            pos2 = nedc_get_pos(montage[i][3], labels_a, params_a[KEYWORD_MMODE])
      else:
            pos2 = int(-1)
      # compute the new length as the shorter of the two
      min_len = len(sig_a[pos1])
      if (pos2 >= int(0)):
            if len(sig_a[pos2]) < min_len:
                min_len = len(sig_a[pos2])
      # copy the first signal
      sig_mont.append(sig_a[pos1])
      sig_mont[i] = sig_mont[i][:min_len]
      # difference the two signals if necessary
      if pos2 >= int(0):
            for j in range(min_len):
                sig_mont[i][j] -= sig_a[pos2][j]
      # append the metadata
      fsamp_mont.append(fsamp_a[pos1])
      labels_mont.append(montage[i][1])
  return (fsamp_mont, sig_mont, labels_mont)


# Signal is resampled to 100Hz
def resampleSignal(original_sample_rate, new_sample_rate, signal_labels, signal_length):
  n = len(signal_labels)
  sig_mont_resampled = [[] for i in range(n)]
  for i in range(n):
    sig_mont_resampled[i] = signal.resample(signal_length[i], int(len(signal_length[i])/original_sample_rate[i]*new_sample_rate))
  return sig_mont_resampled


# Divide each channel into 5s segments
def divideInSegments(segmentLengthInSeconds, frameRate, data):
  channels = len(data)
  segmentLength = segmentLengthInSeconds*frameRate
  totalLength = len(data[0])
  numberOfSegments = int(totalLength/segmentLength)
  sig_mont_seg = np.zeros((channels, numberOfSegments, segmentLength))
  for channel in np.arange(channels):
    channelLength = len(data[channel])
    segment=[]
    segmentNumber=0
    for i in range(channelLength):
      if len(segment)==segmentLength:
        sig_mont_seg[channel][segmentNumber] = segment
        segment=[]
        segmentNumber+=1
      else:
        segment.append(data[channel][i])
  return sig_mont_seg


# This assigns a label to each segment
def assignLabel(startTime, endTime, file, labels, frameRate, segmentLengthInSeconds):
  lbls = labels[file]
  for pos, i in enumerate(lbls):
    lblStart = i[0]*frameRate
    lblEnd = i[1]*frameRate
    lblLbl = i[2]
    if startTime>=lblStart and startTime<lblEnd:
      if endTime<=lblEnd:
        return lblLbl
      else:
        if lblEnd-endTime >= segmentLengthInSeconds/2:
          return lblLbl
        else:
          return lbls[pos+1][2]


#Apply a Short-Time Fourier Transform on each segment of a channel
def applySFTF(data):
  sig_sftf = [[] for i in range(len(data))]
  #Go through each channel
  for i in range(len(data)):
    #Go through each segment in a channel
    for segment in data[i]:
      f, t, Zxx = signal.stft(segment)
      sig_sftf[i].append(Zxx)
  return sig_sftf



################################################################################
# MAIN CODE FORM HERE
################################################################################

def readData(EDF_PATH, EDF_FILE, params, labels):
  SEGMENT_LENGTH = 5
  SAMPLE_RATE = 100

  # Read, resample and segment data from .edf file
  fsamp, sig, ch_labels = nedc_load_edf(EDF_PATH)
  fsamp_sel, sig_sel, labels_sel = nedc_select_channels(params, fsamp, sig, ch_labels)
  fsamp_mont, sig_mont, labels_mont = nedc_apply_montage(params, fsamp_sel, sig_sel, labels_sel)
  sig_mont_resampled = resampleSignal(fsamp_mont, SAMPLE_RATE, labels_mont, sig_mont)
  sig_mont_seg = divideInSegments(SEGMENT_LENGTH, SAMPLE_RATE, sig_mont_resampled)

  numberOfSegments = len(sig_mont_seg[0])
  segmentLength = SEGMENT_LENGTH*SAMPLE_RATE
  # Assign a label to each segment
  sig_labels = ["" for i in range(numberOfSegments)]
  for i in range(numberOfSegments):
    sig_labels[i] = assignLabel(i*segmentLength, (i+1)*segmentLength, EDF_FILE.split('.')[0], labels, SAMPLE_RATE, SEGMENT_LENGTH)

  # Apply Short-Time Fourier Transform to the data
  sig_sftf = applySFTF(sig_mont_seg)

  print(EDF_FILE)

  return sig_sftf, sig_labels

  
