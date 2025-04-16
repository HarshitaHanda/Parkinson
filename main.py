import streamlit as st
import zipfile
import io

# Upload ZIP file containing the dataset
uploaded_file = st.file_uploader("Upload your ZIP file (containing CSV)", type=["zip"])

if uploaded_file is not None:
    # Try to read and extract the ZIP file
    try:
        # Display a message before processing
        st.write("Processing ZIP file...")

        # Try to open the uploaded ZIP file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            # List all files in the ZIP archive
            file_names = zip_ref.namelist()
            st.write("Files in the ZIP archive:")
            st.write(file_names)

            # Try to open the first file in the archive
            csv_file = file_names[0]  # Modify this if the CSV file has a different name

            # Check the contents of the file inside the ZIP
            with zip_ref.open(csv_file) as f:
                # Display the first few lines of the file
                st.write(f.read().decode("utf-8").splitlines()[:5])  # Show the first 5 lines of the file

    except zipfile.BadZipFile:
        st.error("The file you uploaded is not a valid ZIP file or it's corrupted. Please check the file and try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
