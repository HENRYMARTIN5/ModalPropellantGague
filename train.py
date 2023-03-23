import argparse
from betterlib import logging
import sys

log = logging.Logger("./logs/train.log", "train")

if sys.version_info[0] != (3) and sys.version_info[1] < (10):
    log.critical("Python 3.10.x or greater is required to run this program.")
    exit(1)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs to train the model")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training")
ap.add_argument("-o", "--output", type=str, default="model.h5", help="Path to output")
ap.add_argument("-r", "--random", type=bool, default=False, help="Whether to use random data for training (testing purposes only, produces useless model)")
ap.add_argument("-d", "--dataset", type=str, default="dataset/dummy_intuitive_messed.csv", help="Path to dataset if not using random data")
ap.add_argument("-p", "--plot", type=bool, default=False, help="Whether to plot the model")
ap.add_argument("-t", "--threshold", type=float, default=0.1, help="The threshold for the 'Close Enough' metric. The maximum difference between the predicted and actual values for a sample to be considered 'close enough'.")
ap.add_argument("--test-samples", type=int, default=100, help="Number of samples to use for testing, taken from the end of the dataset. Only applies if not using random data")
ap.add_argument("--input-features", type=int, default=4, help="Number of input features")
ap.add_argument("--learning-rate", type=float, default=0.001, help="The learning rate for the Adam optimizer")
args = vars(ap.parse_args())

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import plot_model
from keras.optimizers import Adam
from common import closeenough_metric
import numpy as np

epochs = args["epochs"]
batch_size = args["batch_size"]

if not args["random"]:
    log.info(f"Loading dataset from {args['dataset']}...")
    # Load the data from CSV file
    data = np.genfromtxt(args['dataset'], delimiter=',', skip_header=1)
    # Split the data into training and testing sets
    x_train = data[:-args["test_samples"], 0:args["input_features"]] # First x columns are features for training data
    y_train = data[:-args["test_samples"], args["input_features"]:] # Last column is the target for training data
    x_test = data[-args["test_samples"]:, 0:args["input_features"]]
    y_test = data[-args["test_samples"]:, args["input_features"]:]
else:
    log.warn("Using random data for training! This will produce a useless model, and should only be used for testing.")
    log.info("Creating random dataset...")
    x_train = np.random.rand(100, args["input_features"])
    y_train = np.random.rand(100, 1)
    x_test = np.random.rand(20, args["input_features"])
    y_test = np.random.rand(20, 1)

log.info("Done loading dataset.")
log.debug(f"Training data shape: {x_train.shape}")
log.debug(f"Training labels shape: {y_train.shape}")
log.debug(f"Testing data shape: {x_test.shape}")
log.debug(f"Testing labels shape: {y_test.shape}")

log.info("Creating model...")
# create a Sequential model
model = Sequential()
# Input layer
model.add(Input(shape=(args["input_features"],)))
# Hidden layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and Adam optimizer
log.info("Compiling model...")

model.compile(optimizer=Adam(learning_rate=args["learning_rate"]), loss='binary_crossentropy', metrics=[closeenough_metric(threshold=args['threshold']), 'accuracy'])

log.info(f"Training model for {epochs} epochs with a batch size of {batch_size}...")
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

log.info(f"Saving model to {args['output']}...")
model.save(args['output'])

if args["plot"]:
    log.info("Plotting model...")
    plot_model(model, to_file='img/model_plot.png', show_shapes=True)

log.info("Done.")