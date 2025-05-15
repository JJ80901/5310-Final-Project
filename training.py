#####################################################################################
# training.py
# Author: Josue Cavazos
# Code to train a neural network to predict cold rolled steel (CRS) bends
#####################################################################################

# Import libs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib


# Read local dataset csv into dataframe
crs = pd.read_csv('crs.csv')

# Desired inputs
X = crs[['THK', 'IR']]

# Classification target columns
y_punch = crs['Punch']
y_vdie = crs['VD']

# Create label encorders for punch & v-die
le_punch = LabelEncoder()
le_vdie = LabelEncoder()

# Apply label encoders to each punch & v-die, e.g. 0.03 -> 0 | 0.045 -> 1
y_punch_encoded = le_punch.fit_transform(y_punch)
y_vdie_encoded = le_vdie.fit_transform(y_vdie)

# Normalize the input data (thickness & inside radius) around standard deviation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into a training & testing data set --> 80/20 split
X_train, X_test, y_punch_train, y_punch_test, y_vdie_train, y_vdie_test = train_test_split(
    X_scaled, y_punch_encoded, y_vdie_encoded, test_size = 0.2, random_state = 42)

# Declaire input layer with two features for neural net - thickness & inside radius
input_layer = Input(shape = (X_train.shape[1],))

# Declaire two hidden layers with 32 neurons each with ReLU activation
x = Dense(32, activation = 'relu')(input_layer)
x = Dense(32, activation = 'relu')(x)

# Define output layer with two features, punch & v-die, using softmax activation
output_punch = Dense(len(le_punch.classes_), activation = 'softmax', name = 'punch_output')(x)
output_vdie = Dense(len(le_vdie.classes_), activation = 'softmax', name = 'vdie_output')(x)

# Build mutli-output neural network
model = Model(inputs = input_layer, outputs = [output_punch, output_vdie])

# Prep model for training
model.compile(
    optimizer='adam',
    loss = {
        'punch_output': 'sparse_categorical_crossentropy',
        'vdie_output': 'sparse_categorical_crossentropy'
    },
    metrics = {
        'punch_output': 'accuracy',
        'vdie_output': 'accuracy'
    }
)

# Train model
history = model.fit(
    X_train,
    {'punch_output': y_punch_train, 'vdie_output': y_vdie_train},
    validation_split = 0.2,
    epochs = 100,
    batch_size = 16
)

# Save model & scaler to local directory
model.save("tooling_classifier_model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_punch, "le_punch.pkl")
joblib.dump(le_vdie, "le_vdie.pkl")