# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\Admin\OneDrive\Masaüstü\Machine Learning A-Z\Datasets\Data_red.csv")

# Separate features and target
x = df.drop("quality", axis = 1)
y = df["quality"] #target variable

# Split the dataset into an 80-20 training-test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Create an instance of the StandardScaler class
scaler = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
x_train_scaled = scaler.fit_transform(x_train)

# Apply the transform to the test set
x_test_scaled = scaler.transform(x_test)

# Print the scaled training and test datasets
print(x_train_scaled)
print(x_test_scaled)
