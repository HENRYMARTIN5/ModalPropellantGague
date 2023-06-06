import keras.backend as K

def closeenough_metric(threshold=0.5):
    """
    The "Close Enough" metric. Returns the mean of the number of samples where the model's output is within the threshold (0.5 by default) of the actual value.
    """
    def metric(y_true, y_pred):
        distance = K.abs(y_true - y_pred)
        return K.mean(distance <= threshold)
    return metric

def evaluate_distance(y_true, y_pred):
    """
    Returns the mean distance between the actual and predicted values.
    """
    return K.abs(y_true - y_pred)