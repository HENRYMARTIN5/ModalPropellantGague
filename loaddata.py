import pandas as pd

# Load the CSV file into a pandas dataframe
df = pd.read_csv('dummy.csv', header=None)

# Extract the input features and output labels as NumPy arrays
X_train = df.iloc[:-5, :4].values # First 4 columns as training features
y_train = df.iloc[:-5, 4:].values # Last 5 columns as training labels
X_test = df.iloc[-5:, :4].values # First 4 columns as testing features
y_test = df.iloc[-5:, 4:].values # Last 5 columns as testing labels

# Rename the input feature columns
feature_names = ['train1', 'train2', 'train3', 'train4']
df.columns = feature_names + ['label']

# Print the shapes of the training and testing data
print("Training data shape:")
print(X_train.shape)
print(y_train.shape)
print("Testing data shape:")
print(X_test.shape)
print(y_test.shape)