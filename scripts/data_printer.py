import numpy as np

# Load the big np array files
X = np.load("./formatted_data/mini_train_data.npy")
y = np.load("./formatted_data/mini_train_labels.npy")

print("Data shape -------------------------")
print(X.shape)

print("Labels shape -------------------------")
print(y.shape)
print("Seizures {}".format(np.count_nonzero(y)))
print("Backgrounds {}".format(y.size - np.count_nonzero(y)))
