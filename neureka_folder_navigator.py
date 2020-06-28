################################################################################
# This script needs to be placed into the main data directory (train, dev, etc.)

# What this does:
# 1. Go in one of the three main montage subdirectories, then append their outputs, the result is the whole dataset
# 2. There, starting from '000' go into all folders found up to '134', then append their output
# 3. There, go in all subfolders '00+{current_folder_name}+{all numbers from 00 to 99}', then append their output
# 4. There, go in all subfolders you find, then append their output
# 5. There run the data gathering script on every .edf file found and append their output
# 6. Return that output
################################################################################

import os
import numpy as np
import neureka_data_formatting
from collections import OrderedDict

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

params = {}
labels = {}

################################################################################
# FUNCTIONS
################################################################################

# This reads the parameters file, which specifies the type of montage and which channels to use
def nedc_load_parameters(PARAMS_FILE):
     values = OrderedDict()
     keyword_upcase = KEYWORD_MONTAGE.upper()

     fp = open(PARAMS_FILE, "r")
     # loop over all lines in the file
     flag_pblock = False
     flag_montage = False
     for line in fp:
          str = line.replace(DELIM_SPACE, DELIM_NULL)
          str = str.replace(DELIM_NEWLINE, DELIM_NULL)
          if (str.startswith(DELIM_COMMENT) == True) or (len(str) == 0):
               pass
          # check for the version
          elif str.startswith(KEYWORD_VERSION) == True:
               parts = str.split(DELIM_EQUAL)
               if parts[1] != KEYWORD_VERSION_NUMBER:
                    print("%s (%s: %s): incorrect version number (%s)" %
                          (sys.argv[0], __name__, "nedc_load_parameters",
                           parts[1]))
                    return None
          # check for the beginning or end of a parameter block
          elif (str.startswith(keyword_upcase) == True) and \
               (DELIM_BOPEN in str):
               flag_pblock = True
          elif (flag_pblock == True) and (DELIM_BCLOSE in str):
               fp.close()
               break
          # otherwise, if the parameter block has started, decode a parameter
          elif (flag_pblock == True):
               parts = str.split(DELIM_EQUAL)
               # check for the first occurrence of a montage entry and
               # initialize a list
               if (parts[0] == KEYWORD_MONTAGE) and (flag_montage == False):
                    values[parts[0]] = []
                    flag_montage = True
               # if it is a montage keyword: append the montage list
               if (parts[0] == KEYWORD_MONTAGE):
                    values[parts[0]].append(parts[1].replace(
                         DELIM_QUOTE, DELIM_NULL))
               # else: treat it as a normal name/value pair
               else:
                    values[parts[0]] = parts[1].replace(
                         DELIM_QUOTE, DELIM_NULL)
     fp.close()
     if flag_pblock == False:
          print("%s (%s: %s): invalid parameter file (%s)" %
                (sys.argv[0], __name__, "nedc_load_parameters", pfile_a))
          return None
     return values  

# This loads the labels for each file, with start and end time of any event
def load_labels(lfile):
  all_labels = np.loadtxt(lfile, dtype={'names': ('file', 'start', 'end', 'label', 'confidence'), 'formats': ('U20', 'f4', 'f4', 'U4', 'f4')})
  labels = {}
  for entry in all_labels:
    if entry['file'] in labels.keys():
      labels[entry['file']].append((entry['start'], entry['end'], entry['label']))
    else:
      labels[entry['file']] = [(entry['start'], entry['end'], entry['label'])]
  return labels

# This formats a number into a string with 0s at the beginning
def stringOf(number, length):
  string_of_number = str(number)
  original_length = len(string_of_number)
  for i in range(length-original_length):
    string_of_number = '0'+string_of_number
  return string_of_number

# This merges two datasets together but keeps different channels separate
def appendSegments(a, b):
  if len(a) == 0:
    return b
  for i in range(len(b)):
    a[i].extend(b[i])
  return a

# 5. There run the data gathering script on every .edf file found and append their output
def gatherData(path, file):
  return neureka_data_formatting.readData(path, file, params, labels)

# 4. There, go in all subfolders you find, then append their output
def patientDirectory(patient_code, path):
  patient_data = []
  patient_labels = []
  for (root,dirs,files) in os.walk(path):
    for directory in dirs:
      dir_path = path + '/' + directory
      for (root,dirs,files) in os.walk(dir_path):
        for file in files:
          if file.endswith(".edf"):
            inputs, labels = gatherData(dir_path+'/'+file, file)
            patient_data = appendSegments(patient_data, inputs)
            patient_labels.extend(labels)
  return patient_data, patient_labels

#3. There, go in all subfolders '000+{current_folder_name}+{all numbers from 00 to 99}', then append their output
def idDirectory(id_code, path):
  #Check if durectory 'id_code' exists
  if id_code in os.listdir(path):
    id_data = []
    id_labels = []
    #If yes move into it
    path = path + '/' + id_code
    #Do point 3 above
    for i in range(99):
      patient_code = '000{}{}'.format(id_code, stringOf(i,2))
      if patient_code in os.listdir(path):
        patient_path = path + '/' + patient_code
        inputs, labels = patientDirectory(patient_code, patient_path)
        id_data = appendSegments(id_data, inputs)
        id_labels.extend(labels)
    return id_data, id_labels
  else:
    return None, None

#2. There, starting from '000' go into all folders found up to '134', then print their output to a csv file
def montageDirectory(montage_name, path):
  # Go into montage directory
  montage_path = path + '/' + montage_name
  # Do what stated at point 2. above
  for i in range(135):
    subfolder_name = stringOf(i,3)
    subdata, sublabels = idDirectory(subfolder_name, montage_path)
    if subdata:
      # Append id data to npy files
      try:
        X = np.load("train_data.npy")
        y = np.load("train_labels.npy")

        print(X.shape)
        print(y.shape)

        np.save("train_data.npy", np.append(X, np.asarray(subdata), axis=1))
        np.save("train_labels.npy", np.append(y, np.asarray(sublabels)))
      except IOError:
        np.save("train_data.npy", np.asarray(subdata))
        np.save("train_labels.npy", np.asarray(sublabels))


################################################################################
# MAIN CODE FORM HERE
################################################################################

def main():
  PARAMS_FILE = "params.txt"
  LABELS_FILE = "ref_train.txt"

  #load params file
  global params
  params = nedc_load_parameters(PARAMS_FILE)

  #load labels file
  global labels
  labels = load_labels(LABELS_FILE)

  montages = ["01_tcp_ar", "02_tcp_le", "03_tcp_ar_a"]
  path = os.getcwd()

  montageDirectory(montages[1], path)
  montageDirectory(montages[2], path)

if __name__== "__main__":
  main()
