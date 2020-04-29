import numpy as np

X = np.load("train_data.npy")
print("------ DATA LOADED")
y = np.load("train_labels.npy")
print("------ LABELS LOADED")

cut_X = [[] for i in range(len(X))]
cut_y = []

for i in range(len(X)):
    cut_X[i] = X[i][:144446]

cut_y = y[:144446]


np.save("train_data_02.npy", np.asarray(cut_X))
print("------ DATA SAVED")
np.save("train_labels_02.npy", np.asarray(cut_y))
print("------ LABELS SAVED")