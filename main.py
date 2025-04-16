import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Function to convert string signal data into arrays (not used in this case, but left for completeness)
def convert_to_array(signal_str):
    try:
        return np.array(ast.literal_eval(signal_str.strip('"')))
    except:
        return np.nan

# Streamlit app UI
st.title("Parkinson's Disease Prediction")

st.write("""
    This app predicts Parkinson’s Disease based on your symptoms. Answer the following questions, then click 'Predict' to get the results.
    - Parkinson's Detection
    - Motor vs Non-Motor classification
    - Sleep Disorder prediction
    - Tremor detection
""")

# Simplified inputs for the user (symptoms data)
parkinson_input = st.selectbox("Have you been diagnosed with Parkinson's disease?", ["Yes", "No"])
tremor_input = st.selectbox("Do you experience tremors?", ["Yes", "No"])
sleep_input = st.selectbox("Do you have sleep problems?", ["Yes", "No"])
motor_input = st.selectbox("Do you have motor symptoms like tremors, rigidity, or slow movement?", ["Yes", "No"])
cognitive_input = st.selectbox("Do you experience cognitive symptoms (e.g., memory loss, difficulty concentrating)?", ["Yes", "No"])

# Convert the inputs into numerical values (0 for No, 1 for Yes)
parkinson_val = 1 if parkinson_input == "Yes" else 0
tremor_val = 1 if tremor_input == "Yes" else 0
sleep_val = 1 if sleep_input == "Yes" else 0
motor_val = 1 if motor_input == "Yes" else 0
cognitive_val = 1 if cognitive_input == "Yes" else 0  # New input for cognitive symptoms

# Combine all 5 input features into a single value per timestep
combined_input = (parkinson_val + tremor_val + sleep_val + motor_val + cognitive_val) / 5  # Average the 5 features

# Repeat the combined input across 80 timesteps (to match the expected shape of (80, 1))
X_input = np.array([[combined_input]] * 80)

# Reshape for the model (shape: (1, 80, 1))
X_input = X_input.reshape(1, 80, 1)

# Upload EEG signal image (just for display, not used for prediction)
st.write("### Upload EEG signal image")
uploaded_image = st.file_uploader("Upload EEG Signal Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded EEG Signal", use_column_width=True)

# Button to predict (based on the symptoms input)
if st.button('Predict'):
    # Direct predictions based on simple logic (bypass model)
    def predict_disease(parkinson_val, tremor_val, sleep_val, motor_val, cognitive_val):
        # Parkinson's detection: If the user has Parkinson’s diagnosis or tremors, it's likely positive
        if parkinson_val == 1 or tremor_val == 1:
            parkinson_pred = "Positive"
        else:
            parkinson_pred = "Negative"
        
        # Motor vs Non-Motor: If there are tremors or motor symptoms, classify as motor
        if tremor_val == 1 or motor_val == 1:
            motor_pred = "Motor"
        else:
            motor_pred = "Non-Motor"
        
        # Sleep disorder: If there are sleep problems, predict sleep disorder
        if sleep_val == 1:
            sleep_pred = "Disorder"
        else:
            sleep_pred = "No Disorder"
        
        # Tremor detection: If there are tremors, detect tremors
        if tremor_val == 1:
            tremor_pred = "Detected"
        else:
            tremor_pred = "Not Detected"
        
        return parkinson_pred, motor_pred, sleep_pred, tremor_pred

    # Get predictions using the function
    parkinson_pred, motor_pred, sleep_pred, tremor_pred = predict_disease(parkinson_val, tremor_val, sleep_val, motor_val, cognitive_val)

    # Display the results
    st.write("### Prediction Results:")
    st.write(f"Parkinson's Detection: {parkinson_pred}")
    st.write(f"Motor vs Non-Motor: {motor_pred}")
    st.write(f"Sleep Disorder: {sleep_pred}")
    st.write(f"Tremor Detection: {tremor_pred}")
