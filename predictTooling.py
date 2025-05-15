#####################################################################################
# predictTooling.py
# Author: Josue Cavazos
# Code to use a neural network to predict cold rolled steel (CRS) bends
#####################################################################################

# Import libs
import numpy as np
import tensorflow as tf
import joblib

# Load local model & scalar
model = tf.keras.models.load_model("tooling_classifier_model.h5")
scaler = joblib.load("scaler.pkl")
le_punch = joblib.load("le_punch.pkl")
le_vdie = joblib.load("le_vdie.pkl")

# User input with examples
thickness = input("Thickness: ") #0.045 -> example thickness
inside_radius = input("Desired Inside Radius: ") #0.060 -> example inside radius

# Prepare input for prediction
X_input = np.array([[thickness, inside_radius]])
X_scaled = scaler.transform(X_input)

# Make prediction with model
punch_pred, vdie_pred = model.predict(X_scaled)

# Get the prediction with highest probability
punch_class = np.argmax(punch_pred, axis=1)
vdie_class = np.argmax(vdie_pred, axis=1)

# Decode values back to original sizes
punch_size = le_punch.inverse_transform(punch_class)[0]
vdie_size = le_vdie.inverse_transform(vdie_class)[0]

# Print result in terminal
print(f"Recommended Punch Size: {punch_size}")
print(f"Recommended V-Die Size: {vdie_size}")