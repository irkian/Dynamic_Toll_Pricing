import streamlit as st
import pandas as pd
import numpy as np
import pickle



# Add custom CSS to hide the GitHub icon
st.markdown(
    
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    ,
    unsafe_allow_html=True
)

# Load the model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Function to make predictions with original values
def predict_with_original_values(direction, start_point, end_point, traffic_volume, hour, day_of_week):
    # Encode the input values
    direction_encoded = encoders['direction'].transform([direction])[0]
    start_point_encoded = encoders['start_point'].transform([start_point])[0]
    end_point_encoded = encoders['end_point'].transform([end_point])[0]

    # Create the input array
    input_array = [[direction_encoded, start_point_encoded, end_point_encoded, traffic_volume, hour, day_of_week]]


    # Make prediction
    prediction = model.predict(input_array)
    return prediction[0]

# Function to adjust rate based on traffic
def adjust_rate_based_on_traffic(predicted_rate, traffic_volume, threshold=400):
    if (traffic_volume > threshold):
        adjusted_rate = predicted_rate * (1 + (traffic_volume - threshold) / threshold)
    else:
        adjusted_rate = predicted_rate * (1 - (threshold - traffic_volume) / threshold)
    return adjusted_rate

# Streamlit App
st.title("Dynamic Toll Price Prediction")

# User inputs
direction = st.selectbox('Direction', options=encoders['direction'].classes_)
start_point = st.selectbox('Start Point', options=encoders['start_point'].classes_)
end_point = st.selectbox('End Point', options=encoders['end_point'].classes_)
day_of_week = st.selectbox('Day of the Week', options=list(range(7)))

# Generate traffic volume and hour
time_hours = np.arange(1, 25)
traffic_volumes = np.linspace(100, 1000, num=24)

# Placeholder for the results
st.subheader("Predicted Rates Over 24 Hours")
for hour, traffic_volume in zip(time_hours, traffic_volumes):
    predicted_rate = predict_with_original_values(direction, start_point, end_point, int(traffic_volume), hour, day_of_week)
    adjusted_rate = adjust_rate_based_on_traffic(predicted_rate, int(traffic_volume))
    st.write(f"Hour: {hour}, Traffic Volume: {int(traffic_volume)}, Predicted Rate: {round(predicted_rate, 2)}, Adjusted Rate: {round(adjusted_rate, 2)}")
