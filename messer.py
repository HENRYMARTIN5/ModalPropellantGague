"""
This script goes through the dataset and messes up each of the numbers by a random amount.
For example, if the number is 0.5, it could be changed to 0.4 or 0.6. Or, even, 0.4142860.
"""
import argparse
import csv
import random
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="dataset/dummy_intuitive.csv", help="Path to evaluation dataset")
ap.add_argument("-o", "--output", type=str, default="dataset/dummy_intuitive_messed.csv", help="Path to output dataset")
ap.add_argument("-m", "--max-mess", type=float, default=0.1, help="The maximum amount to mess up each number by.")
args = vars(ap.parse_args())

# Load the data from CSV file
data = np.genfromtxt(args['dataset'], delimiter=',', skip_header=1)
x_test = data[:, 0:4]
y_test = data[:, 4:]

# Mess up the data
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        x_test[i][j] += random.uniform(-args['max_mess'], args['max_mess'])
for i in range(len(y_test)):
    for j in range(len(y_test[i])):
        y_test[i][j] += random.uniform(-args['max_mess'], args['max_mess'])

# Check if any values are out of range
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        if x_test[i][j] < 0:
            x_test[i][j] = 0
for i in range(len(y_test)):
    for j in range(len(y_test[i])):
        if y_test[i][j] < 0:
            y_test[i][j] = 0
        elif y_test[i][j] > 1:
            y_test[i][j] = 1

# Write the data to a new CSV file
with open(args['output'], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['train1', 'train2', 'train3', 'train4', 'label'])
    for i in range(len(x_test)):
        writer.writerow(np.append(x_test[i], y_test[i]))