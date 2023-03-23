import argparse
from betterlib import logging
import sys

log = logging.Logger("./logs/eval.log", "eval")

if sys.version_info[0] != (3) and sys.version_info[1] < (10):
    log.critical("Python 3.10.x or greater is required to run this program.")
    exit(1)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# positional arguments
ap.add_argument("-d", "--dataset", type=str, default="dataset/dummy_intuitive_messed.csv", help="Path to evaluation dataset")
ap.add_argument("-m", "--model", type=str, default="model.h5", help="Path to pre-trained model")
ap.add_argument("-t", "--threshold", type=float, default=0.1, help="The threshold for the 'Close Enough' metric. The maximum difference between the predicted and actual values for a sample to be considered 'close enough'.")
ap.add_argument("--input-features", type=int, default=4, help="Number of input features")
args = vars(ap.parse_args())

from keras.models import load_model
from common import closeenough_metric, evaluate_distance
import numpy as np

log.info(f"Loading model from {args['model']}...")
model = load_model(args['model'], custom_objects={"metric": closeenough_metric})

log.info(f"Loading dataset from {args['dataset']}...")
# Load the data from CSV file
data = np.genfromtxt(args['dataset'], delimiter=',', skip_header=1)
x_test = data[:, 0:args["input_features"]]
y_test = data[:, args["input_features"]:]

log.info("Done loading model and dataset.")
log.debug(f"Testing data shape: {x_test.shape}")
log.debug(f"Testing labels shape: {y_test.shape}")

log.info("Evaluating model...")
# Loop through each sample and evaluate it
overall_accuracy = 0
average_distance = 0
for i in range(len(x_test)):
    sample = x_test[i]
    actual = y_test[i]
    predicted = model.predict(np.array([sample]))[0]
    overall_accuracy += closeenough_metric(threshold=args['threshold'])(actual, predicted)
    average_distance += evaluate_distance(actual, predicted)
    log.debug(f"Sample {i}: Predicted {predicted}, Actual {actual}, Distance: {evaluate_distance(actual, predicted)} Close Enough: {closeenough_metric(threshold=args['threshold'])(actual, predicted)}")

log.info(f"Overall accuracy: {overall_accuracy / len(x_test) * 100}% on {len(x_test)} samples. (Threshold: {args['threshold']})")
log.info(f"Average distance: {average_distance / len(x_test)}.")