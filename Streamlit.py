import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import warnings

# Suppress all Python warnings
warnings.filterwarnings("ignore")

# Set up the page configuration and layout
st.set_page_config(page_title='Instagram Fake Profile Detector', layout='wide')

# Function to load the model
def load_my_model():
    return load_model("instagram_fake_detection_model.h5")

# Load model immediately for use in the app
model = load_my_model()

# App title and description
st.title('Instagram Fake Profile Detection')
st.markdown("""
This tool helps detect fake Instagram profiles based on various metrics. Upload an Excel file with the necessary data.
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose a file (in .xlsx format)", type="xlsx")
if uploaded_file is not None:
    with st.spinner('Loading data...'):
        try:
            data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    # Define and process features
    data['has_profile_pic'] = data['profilePicUrl'].apply(lambda x: 0 if pd.isna(x) else 1)
    data['username_num_ratio'] = data['username'].apply(lambda x: sum(c.isdigit() for c in str(x)) / len(x) if pd.notna(x) and len(x) > 0 else 0)
    data['fullname_word_count'] = data['fullName'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    data['fullname_num_ratio'] = data['fullName'].apply(lambda x: sum(c.isdigit() for c in str(x)) / len(x) if pd.notna(x) and len(x) > 0 else 0)
    data['name_is_username'] = (data['username'] == data['fullName']).astype(int)
    data['bio_length'] = data['biography'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    data['is_private'] = data['private'].astype(bool).astype(int)
    data['followers_following_ratio'] = data.apply(lambda row: row['followersCount'] / (row['followsCount'] + 1) if row['followsCount'] + 1 > 0 else 0, axis=1)

    # Prepare features for prediction
    feature_columns = [
        'has_profile_pic', 'username_num_ratio', 'fullname_word_count',
        'fullname_num_ratio', 'name_is_username', 'bio_length',
        'is_private', 'postsCount', 'followersCount', 'followers_following_ratio'
    ]
    features = data[feature_columns].copy()
    features.fillna(0, inplace=True)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Make predictions
    predictions = model.predict(features_scaled)
    data['fake_score'] = predictions.flatten()
    data['is_fake'] = ((data['fake_score'] > 0.02) & (data['followers_following_ratio'] < 0.2)) | ((data['followersCount'] < 50) & (data['followsCount'] < 50))

    # Handle NaN values before converting to integer
    data['followersCount'] = data['followersCount'].fillna(0).astype(int)
    data['followsCount'] = data['followsCount'].fillna(0).astype(int)
    
    # Selecting profiles to display
    fake_profiles = data[data['is_fake']][['username', 'url', 'fullName', 'followersCount', 'followsCount', 'is_fake']]

    if not fake_profiles.empty:
        st.success('Analysis complete! Detected Fake Profiles:')
        # Display fake profiles with the ability to display profile images
        st.write(fake_profiles.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info('No fake profiles detected.')
else:
    st.info('Please upload an Excel file to begin analysis.')

# Additional features and footer
st.markdown('---')
st.markdown('Developed by | © 2025 All rights reserved')
