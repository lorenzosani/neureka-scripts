import numpy as np

# Load the big np array files
X = np.load("./formatted_data/train_data_02.npy")
print("------ DATA LOADED")
y = np.load("./formatted_data/train_labels_02.npy")
print("------ LABELS LOADED")

# Create empty mini np arrays
mini_X = np.zeros((20,10000,129,5))
mini_y = np.zeros(10000)

# Initialise counters
i = 0
j = 0
seizures = 0
backgrounds = 0

# Find 5000 segments with seizures and 5000
while seizures < 5000 and i < y.size:
    if y[j] == 'seiz':
        # If a segment is a seizure, load it into the new arrays and increase the counters
        mini_y[i] = 1.0
        for channel in range(len(X)):
            mini_X[channel, i] = X[channel, j]
        seizures += 1
        i += 1
        j += 1
    elif backgrounds < 5000:
        # If a segment is a background, add it to the new arrays if there's space left and increase the counters
        mini_y[i] = 0.0
        for channel in range(len(X)):
            mini_X[channel, i] = X[channel, j]
        backgrounds += 1
        i += 1
        j += 1
    else:
        j += 1

# Save the mini arrays in a npy file
np.save("./formatted_data/mini_train_data.npy", mini_X)
np.save("./formatted_data/mini_train_labels.npy", mini_y)
    





