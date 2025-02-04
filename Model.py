import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load data
data_path = "Chelappan_Following.xlsx"
data = pd.read_excel(data_path)

# Define features based on provided criteria
data['has_profile_pic'] = data['profilePicUrl'].apply(lambda x: 0 if pd.isna(x) else 1)
data['username_num_ratio'] = data['username'].apply(lambda x: sum(c.isdigit() for c in x) / len(x) if pd.notna(x) and len(x) > 0 else 0)
data['fullname_word_count'] = data['fullName'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
data['fullname_num_ratio'] = data['fullName'].apply(lambda x: sum(c.isdigit() for c in x) / len(x) if pd.notna(x) and len(x) > 0 else 0)
data['name_is_username'] = (data['username'] == data['fullName']).astype(int)
data['bio_length'] = data['biography'].apply(lambda x: len(x) if pd.notna(x) else 0)
data['is_private'] = data['private'].fillna(False).astype(int)
data['followers_following_ratio'] = data.apply(lambda row: row['followersCount'] / (row['followsCount'] + 1), axis=1)

# Define Labels based on criteria
data['label'] = ((data['postsCount'] < 10) & (data['followers_following_ratio'] < 0.5)).astype(int)

# Select and prepare features for modeling
feature_columns = [
    'has_profile_pic', 'username_num_ratio', 'fullname_word_count',
    'fullname_num_ratio', 'name_is_username', 'bio_length',
    'is_private', 'postsCount', 'followersCount', 'followers_following_ratio'
]
features = data[feature_columns]

# Handle missing values and scale features
features.fillna(0, inplace=True)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, data['label'], test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# Save the trained model
model.save("instagram_fake_detection_model.h5")
print("Model saved as instagram_fake_detection_model.h5")
