import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import time
import os
import warnings

warnings.filterwarnings('ignore')

# ---------------- USER AUTH SETUP ---------------- #

# Load user credentials from file if exists
if os.path.exists("users.pkl"):
    with open("users.pkl", "rb") as f:
        user_credentials = pickle.load(f)
else:
    user_credentials = {
        'credentials': {
            'usernames': {
                'user1': {
                    'name': 'User One',
                    'password': '$2b$12$0GXcWVrFZBA1t5gMVOv9NeiEEtKAnEo5yDlRUctzMQBWGSK0qzjya'  # hashed password: user1pass
                },
                'user2': {
                    'name': 'User Two',
                    'password': '$2b$12$5A5w8A0.RqM3vg2q9jNHSeDN1F1pHTcI6H1DWaNwr69jXWZ3hC2fq'  # hashed password: user2pass
                }
            }
        }
    }

# Save user credentials
def save_users():
    with open("users.pkl", "wb") as f:
        pickle.dump(user_credentials, f)

# Function to add new user
def add_user(username, password):
    hashed_password = stauth.Hasher([password]).generate()[0]
    user_credentials['credentials']['usernames'][username] = {
        'name': username,
        'password': hashed_password
    }
    save_users()

# Authenticator setup
authenticator = stauth.Authenticate(
    user_credentials['credentials'],
    "stress_auth", 
    "random_signature_key", 
    cookie_expiry_days=1
)

# ---------------- MAIN APP ---------------- #

login_result = authenticator.login('Login', location='main')

if login_result:
    name, authentication_status, username = login_result
    if authentication_status:
        st.success(f"Welcome {name}!")
        authenticator.logout('Logout', 'sidebar')

        st.title("🧠 Human Stress Detection")
        st.write("An interactive web app to detect stress level based on sleeping habits.")

        # Load dataset
        try:
            df = pd.read_csv('stress.csv')
            # Train model
            X = df[['snoring_rate', 'respiration_rate', 'body_temp', 'limb_movement', 'blood_oxygen',
                    'eye_movement', 'sleep_hours', 'heart_rate']]
            y = df['stress_level']
            model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=5, min_samples_leaf=10,
                                           min_samples_split=10, min_weight_fraction_leaf=0.1)
            model.fit(X, y)

            with open('trained_model.sav', 'wb') as f:
                pickle.dump(model, f)

            loaded_model = pickle.load(open('trained_model.sav', 'rb'))

            def predictor(sr, rr, t, lm, bo, rem, sh, hr):
                prediction = loaded_model.predict(np.array([[sr, rr, t, lm, bo, rem, sh, hr]]))
                return prediction

            def level(n):
                return {
                    0: "Low / Normal",
                    1: "Medium Low",
                    2: "Medium",
                    3: "Medium High",
                    4: "High"
                }.get(n, "")

            option = st.radio("How would you like to give input?", ('Slider', 'Fill Values'))

            if option == 'Fill Values':
                sr = st.number_input('Snoring Rate (dB)')
                rr = st.number_input('Respiration Rate (breaths per minute)')
                t = st.number_input('Body Temperature (°F)')
                lm = st.number_input('Limb Movement')
                bo = st.number_input('Blood Oxygen')
                rem = st.number_input('Eye Movement')
                sh = st.number_input('Sleeping Hours (hr)')
                hr = st.number_input('Heart Rate (bpm)')
            else:
                sr = st.slider('Snoring Rate (dB)', 45, 100, 71)
                rr = st.slider('Respiration Rate (breaths per minute)', 15, 30, 21)
                t = st.slider('Body Temperature (°F)', 85, 110, 92)
                lm = st.slider('Limb Movement', 4, 20, 11)
                bo = st.slider('Blood Oxygen', 80, 100, 90)
                rem = st.slider('Eye Movement', 60, 105, 88)
                sh = st.slider('Sleeping Hours (hr)', 0, 12, 4)
                hr = st.slider('Heart Rate (bpm)', 50, 100, 64)

            if st.button("RUN"):
                with st.spinner(text='Predicting...'):
                    time.sleep(2)
                prediction = predictor(sr, rr, t, lm, bo, rem, sh, hr)
                st.write("You entered:", sr, rr, t, lm, bo, rem, sh, hr)
                st.success(f"Stress Level: {prediction[0]} - {level(int(prediction))}")

            with st.expander("📄 Dataset & Model Info"):
                st.caption("Dataset Columns:")
                st.code('''
sr - snoring rate
rr - respiration rate
t - body temperature
lm - limb movement
bo - blood oxygen
rem - eye movement
sh - sleeping hours
hr - heart rate
sl - stress level
                ''')
                st.write("First 5 rows of the dataset:")
                st.write(df.head())
                st.write("Model used:")
                st.code("DecisionTreeClassifier(max_depth=5, max_leaf_nodes=5, min_samples_leaf=10, min_samples_split=10, min_weight_fraction_leaf=0.1)")
                st.caption("Training Accuracy: 0.998")
                st.caption("Validation Accuracy: 0.992")
                try:
                    st.image(Image.open('CM.jpeg'), caption='Confusion Matrix')
                    st.image(Image.open('DT.png'), caption='Decision Tree')
                except:
                    st.warning("Confusion Matrix (CM.jpeg) or Decision Tree (DT.png) images not found.")

        except FileNotFoundError:
            st.error("❌ File 'stress.csv' not found. Please provide the correct path.")

    elif authentication_status is False:
        st.error("Username or password is incorrect.")
    elif authentication_status is None:
        st.warning("Please enter your username and password.")

# ---------------- REGISTRATION ---------------- #

if authentication_status is False or authentication_status is None:
    with st.expander("🆕 Register New User"):
        new_username = st.text_input('Choose a username:')
        new_password = st.text_input('Choose a password:', type='password')

        if st.button("Register"):
            if new_username in user_credentials['credentials']['usernames']:
                st.error("Username already exists.")
            elif new_username == "" or new_password == "":
                st.warning("Please fill out both fields.")
            else:
                add_user(new_username, new_password)
                st.success("Registration successful! You can now log in.")