import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import ast
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Load the trained model
model = load_model('parkinson_multi_task_model.h5')

# Function to convert string signal data into arrays
def convert_to_array(signal_str):
    try:
        return np.array(ast.literal_eval(signal_str.strip('"')))
    except:
        return np.nan

# Streamlit app UI
st.title("Parkinson's Disease Prediction")

st.write("""
    This app predicts Parkinson’s Disease based on your symptoms. Upload a ZIP file containing your dataset, and the app will predict:
    - Parkinson's Detection
    - Motor vs Non-Motor classification
    - Sleep Disorder prediction
    - Tremor detection
""")

# Upload ZIP file containing the dataset
uploaded_file = st.file_uploader("Upload your ZIP file (containing CSV)", type=["zip"])

if uploaded_file is not None:
    # Try to read and extract the ZIP file
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            # List all files in the ZIP archive
            file_names = zip_ref.namelist()

            # Assuming the first file is the CSV file
            csv_file = file_names[0]  # Modify this if needed

            # Read the CSV file from the ZIP
            with zip_ref.open(csv_file) as f:
                df = pd.read_csv(f)

            # Preprocess the uploaded data
            df['Raw_Signal'] = df['Raw_Signal'].apply(convert_to_array)
            df['Magnitude'] = df['Magnitude'].apply(convert_to_array)
            df['Phase'] = df['Phase'].apply(convert_to_array)

            # Clean data by dropping rows with NaN values in key columns
            df = df.dropna(subset=['Raw_Signal', 'Magnitude', 'Phase'])

            # Normalize Raw_Signal
            scaler = StandardScaler()
            raw_signals = np.vstack(df['Raw_Signal'].values)
            raw_signals_normalized = scaler.fit_transform(raw_signals)

            # Pad/truncate to 80 timesteps
            X_padded = pad_sequences(raw_signals_normalized, maxlen=80, padding='post', truncating='post')
            X = X_padded.reshape(len(df), 80, 1)

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

    except zipfile.BadZipFile:
        st.error("The file you uploaded is not a valid ZIP file or it's corrupted. Please check the file and try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
