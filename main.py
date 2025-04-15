import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('parkinson_multi_task_model.h5')

# Streamlit app UI
st.title("Parkinson's Disease Prediction")

st.write("""
    This app predicts Parkinson’s Disease based on your symptoms. Answer a few simple questions below, and the app will predict:
    - Parkinson's Detection
    - Motor vs Non-Motor classification
    - Sleep Disorder prediction
    - Tremor detection
""")

# Simplified inputs for the user
parkinson_input = st.selectbox("Have you been diagnosed with Parkinson's disease?", ["Yes", "No"])
tremor_input = st.selectbox("Do you experience tremors?", ["Yes", "No"])
sleep_input = st.selectbox("Do you have sleep problems?", ["Yes", "No"])
motor_input = st.selectbox("Do you have motor symptoms like tremors, rigidity, or slow movement?", ["Yes", "No"])

# Convert the inputs into numerical values (0 for No, 1 for Yes)
parkinson_val = 1 if parkinson_input == "Yes" else 0
tremor_val = 1 if tremor_input == "Yes" else 0
sleep_val = 1 if sleep_input == "Yes" else 0
motor_val = 1 if motor_input == "Yes" else 0

# Create an input array for the model (you can adjust this structure as needed)
# We are passing the simplified symptom data as an array
X_input = np.array([[parkinson_val, tremor_val, sleep_val, motor_val]])

if st.button('Predict'):
    # Get predictions from the model
    predictions = model.predict(X_input)

    # Unpack predictions
    y_parkinson_pred = predictions[0]
    y_motor_pred = predictions[1]
    y_sleep_pred = predictions[2]
    y_tremor_pred = predictions[3]

    # Display the results
    st.write("### Prediction Results:")
    st.write(f"Parkinson's Detection: {'Positive' if y_parkinson_pred[0] == 1 else 'Negative'}")
    st.write(f"Motor vs Non-Motor: {'Motor' if y_motor_pred[0] == 1 else 'Non-Motor'}")
    st.write(f"Sleep Disorder: {'Disorder' if y_sleep_pred[0] == 1 else 'No Disorder'}")
    st.write(f"Tremor Detection: {'Detected' if y_tremor_pred[0] == 1 else 'Not Detected'}")
