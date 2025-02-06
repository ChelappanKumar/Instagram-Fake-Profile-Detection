import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

### Criteria for Fake Profile Detection:
- **Profile Picture Availability:** Fake profiles often lack a profile picture.
- **Username-to-Number Ratio:** Fake accounts may have usernames with many numbers (e.g., john12345).
- **Full Name Word Count:** Genuine profiles typically have meaningful names; fake ones may not.
- **Full Name Numeric Ratio:** Fake profiles sometimes include numbers in their full name.
- **Name Matches Username:** Identical full names and usernames are rare for real users but common in fake accounts.
- **Biography Length:** Fake profiles often have very short or empty bios.
- **Privacy Status:** Some fake accounts are private to avoid scrutiny.
- **Number of Posts:** Fake profiles often have very few or no posts.
- **Number of Followers:** Fake accounts typically have abnormally low or artificially high follower counts.
- **Followers-to-Following Ratio:** Fake accounts tend to follow many users but have few followers in return.
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
    
    # Assign classification labels
    data['is_fake'] = np.where(
        ((data['fake_score'] > 0.02) & (data['followers_following_ratio'] < 0.2)) | 
        ((data['followersCount'] < 50) & (data['followsCount'] < 50)),
        'Fake',
        'Real'
    )
    
    data['warning'] = (
        ((data['fake_score'] > 0.015) & (data['followers_following_ratio'] < 0.02)) | 
        (((data['followersCount'] < 100) & (data['followersCount'] > 50)) & (data['followsCount'] < 50))
    )

    data.loc[data['warning'], 'is_fake'] = 'Warning'
    
    # Add new column with emojis
    data['alert'] = data['is_fake'].apply(lambda x: '🚨' if x == 'Fake' else ('⚠️' if x == 'Warning' else '✅ Real'))

    # Create explanations for why a profile is Fake or Warning
    def get_reason(row):
        reasons = []
        if row['is_fake'] == 'Fake':
            if row['fake_score'] > 0.02:
                reasons.append("High fake score")
            if row['followers_following_ratio'] < 0.2:
                reasons.append("Low followers/following ratio")
            if row['followersCount'] < 50 and row['followsCount'] < 50:
                reasons.append("Very low followers and follows count")
        elif row['is_fake'] == 'Warning':
            if row['fake_score'] > 0.015:
                reasons.append("Moderate fake score")
            if row['followers_following_ratio'] < 0.02:
                reasons.append("Extremely low followers/following ratio")
            if 50 < row['followersCount'] < 100 and row['followsCount'] < 50:
                reasons.append("Suspicious follower/following pattern")
        return ", ".join(reasons)

    data['Action'] = data.apply(get_reason, axis=1)

    # Selecting profiles to display
    fake_profiles = data[data['is_fake'] != 'Real'][['username', 'url', 'fullName', 'followersCount', 'followsCount', 'is_fake', 'alert', 'Action']]

    if not fake_profiles.empty:
        st.success('Analysis complete! Detected Fake and Warning Profiles:')
        st.write("### Detected Profiles:")
        st.dataframe(fake_profiles, use_container_width=True)
    else:
        st.info('No fake or suspicious profiles detected.')

else:
    st.info('Please upload an Excel file to begin analysis.')

# Additional features and footer
st.markdown('---')
st.markdown('Developed by | © 2025 All rights reserved')
