import argparse
from betterlib import logging

log = logging.Logger("./logs/train.log", "train")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs to train the model")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training")
ap.add_argument("-o", "--output", type=str, default="model.h5", help="Path to output")
ap.add_argument("-r", "--random", type=bool, default=False, help="Whether to use random data for training (testing purposes only, produces useless model)")
ap.add_argument("-p", "--plot", type=bool, default=False, help="Whether to plot the model")
args = vars(ap.parse_args())

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import plot_model
import numpy as np

epochs = args["epochs"]
batch_size = args["batch_size"]

if not args["random"]:
    log.critical("Dataset loading is not yet implemented! Please use the `-r true` flag for now.")
    exit()
    # Eventually, a dataset will be loaded here.
    # The first 16 features will be used to pass the initial state of the frequencies,
    # and the last 16 features will be used to pass the current state. That way, the
    # model will be able to estimate with some accuracy the percentage of fuel left.
else:
    log.warn("Using random data for training! This will produce a useless model, and should only be used for testing.")
    log.info("Creating random dataset...")
    x_train = np.random.rand(100, 32) # create random input data with 100 samples and 32 features
    y_train = np.random.rand(0, 2, size=(100, 1)) # create random output data with 100 samples
    x_test = np.random.rand(20, 32) # create random input data with 20 samples for testing
    y_test = np.random.rand(0, 2, size=(20, 1)) # create random output data with 20 samples for testing

log.info("Creating model...")
# create a Sequential model
model = Sequential()
# Input layer
model.add(Input(shape=(32,))) # 32 input features
# Hidden layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and Adam optimizer
log.info("Compiling model...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

log.info(f"Training model for {epochs} epochs with a batch size of {batch_size}...")
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

log.info(f"Saving model to {args['output']}...")
model.save(args['output'])

if args["plot"]:
    log.info("Plotting model...")
    plot_model(model, to_file='model.png', show_shapes=True)

log.info("Done.")