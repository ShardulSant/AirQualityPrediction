from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import rasterio
from rasterio.transform import from_bounds
from pyproj import Proj, transform
from flask_cors import CORS
from datetime import datetime
import requests
import os


import uuid


@tf.keras.utils.register_keras_serializable()
def grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)


app = Flask(__name__)
CORS(app)
tf.keras.config.enable_unsafe_deserialization()
# Load the pre-trained models and scalers
model = tf.keras.models.load_model('model/best_combined_model (1).keras', custom_objects={"mse": "mean_squared_error",'grayscale_to_rgb': grayscale_to_rgb})
scaler = joblib.load("model/scaler_meteor4.pkl")
encoder = joblib.load("model/temporal_encoder_final4.pkl")
scaler_target = joblib.load("model/scaler_target_final4.pkl")

# Function to load, crop, preprocess, and save .tif images based on latitude and longitude
def load_and_crop_image(img_path, latitude, longitude, crop_size=120, buffer_distance=1000, channels=1, img_name="image"):
    with rasterio.open(img_path) as src:
        # Convert the latitude and longitude to the image's coordinate system
        lonlat_proj = Proj(init='epsg:4326')  # WGS84 coordinate system
        if channels == 3:
            img_proj = Proj(init='epsg:32643')
        else:
            img_proj = Proj(src.crs)              # Get the image's CRS
        x, y = transform(lonlat_proj, img_proj, longitude, latitude)

        # Calculate bounding box for cropping
        buffer_meters = buffer_distance
        left = x - buffer_meters
        right = x + buffer_meters
        bottom = y - buffer_meters
        top = y + buffer_meters

        # Crop the image to the bounding box
        window = src.window(left, bottom, right, top)
        
        # Read as RGB or grayscale depending on `channels`
        if channels == 3:
            cropped_img = np.dstack([src.read(i, window=window) for i in (1, 2, 3)])  # Read RGB channels
        else:
            cropped_img = src.read(1, window=window)  # Read as grayscale
        
        # Resize and normalize to the model input size
        cropped_img = cropped_img / 255.0
        cropped_img_resized = np.resize(cropped_img, (crop_size, crop_size, channels))
        
        # Save the cropped image
        save_path = f"cropped_images/{img_name}_{uuid.uuid4().hex}.tif"
        with rasterio.open(
            save_path, 'w', driver='GTiff', height=crop_size, width=crop_size,
            count=channels, dtype=cropped_img_resized.dtype
        ) as dst:
            if channels == 3:
                for i in range(1, 4):
                    dst.write(cropped_img_resized[:, :, i - 1], i)
            else:
                dst.write(cropped_img_resized[:, :, 0], 1)

    return cropped_img_resized.reshape((1, crop_size, crop_size, channels))

# Function to retrieve latest weather data
def latest_weather_data(lat, lon):
    api_key = "52e4c34c4d24b0323f07c3ae892a30b5"
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print(data)
        weather_data = {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"]["deg"],
            "timestamp": datetime.utcfromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S'),
	        "data":data
        }
        return weather_data
    else:
        print("Failed to retrieve data:", response.status_code, response.text)
        return None
def calculate_aqi(concentration, breakpoints):
    """
    Calculate AQI sub-index for a pollutant based on its concentration and breakpoints.
    
    Args:
    - concentration (float): The measured pollutant concentration.
    - breakpoints (list of tuples): Breakpoints in the format [(BP_lo, BP_hi, I_lo, I_hi), ...].
    
    Returns:
    - int: AQI sub-index for the pollutant.
    """
    for bp in breakpoints:
        BP_lo, BP_hi, I_lo, I_hi = bp
        if BP_lo <= concentration <= BP_hi:
            return round((I_hi - I_lo) / (BP_hi - BP_lo) * (concentration - BP_lo) + I_lo)
    return None  # If concentration is out of range

# Define breakpoints for each pollutant (Indian AQI System)
breakpoints = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300),
              (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300),
             (351, 430, 301, 400), (431, 600, 401, 500)],
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300),
            (801, 1600, 301, 400), (1601, 3200, 401, 500)],
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300),
            (281, 400, 301, 400), (401, 500, 401, 500)],
    "CO": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300),
           (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "Ozone": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300),
              (209, 748, 301, 400), (749, 1000, 401, 500)]
}
def classify_aqi(aqi):
    """
    Classify the AQI value into a category based on the Indian AQI system.
    
    Args:
    - aqi (int): The calculated AQI value.
    
    Returns:
    - str: The AQI category and description.
    """
    if 0 <= aqi <= 50:
        return "Good"
    elif 51 <= aqi <= 100:
        return "Satisfactory"
    elif 101 <= aqi <= 200:
        return "Moderate"
    elif 201 <= aqi <= 300:
        return "Poor"
    elif 301 <= aqi <= 400:
        return "Very Poor"
    elif 401 <= aqi <= 500:
        return "Severe"
    else:
        return "Invalid AQI value."

# Route for home page
@app.route('/')
def hello_world():
    return 'Air Quality Prediction API is running!'

# Route to handle air quality prediction requests
@app.route('/predict', methods=['POST'])
def predict_air_quality():
    data = request.json
    lat = data['lat']
    lon = data['lon']
    met_data = latest_weather_data(lat, lon)
    d = pd.read_excel("airqualityprediction.xlsx")
    t = d[['Day_of_Week', 'Hour_of_Day', 'Month']].values
    temp = encoder.transform(t)
    m = d[['T2M', 'RH2M', 'WS10M', 'WD10M']].values

    met = scaler.transform(m)
    # Step 1: Preprocess Meteorological Data
    meteorological_data = np.array([[met_data['temperature'], met_data['humidity'], met_data['wind_speed'], met_data['wind_direction']]])
    meteorological_data_scaled = scaler.transform(meteorological_data)

    # Step 2: Extract and preprocess Temporal Data
    timestamp = datetime.strptime(met_data['timestamp'], '%Y-%m-%d %H:%M:%S')
    temporal_data = np.array([[timestamp.weekday(), timestamp.hour, timestamp.month]])
    temporal_data_encoded = encoder.transform(temporal_data)

    # Step 3: Crop and preprocess NDVI, Urbanization, and RGB Surface Reflectance images
    image_folder = "satelliteimagery"  # Replace with the actual path to images
    ndvi_image = load_and_crop_image(os.path.join(image_folder, "ndvi_10m.tif"), lat, lon, img_name="ndvi")
    urban_image = load_and_crop_image(os.path.join(image_folder, "urban_index_10m.tif"), lat, lon, img_name="urban")
    lsr_image = load_and_crop_image(os.path.join(image_folder, "B11_resampled_10m.tif"), lat, lon, img_name="lsr")  # RGB image

    # Step 4: Make prediction using the model
    predicted_air_quality = model.predict([meteorological_data_scaled, temporal_data_encoded, ndvi_image, urban_image, lsr_image])
    predicted_air_quality = scaler_target.inverse_transform(predicted_air_quality)
    
    # Step 5: Prepare and return the prediction response
    response = {
        "PM2.5": abs(float(predicted_air_quality[0][0])),
        "PM10": abs(float(predicted_air_quality[0][1])),
        "NO": abs(float(predicted_air_quality[0][2])),
        "NO2": abs(float(predicted_air_quality[0][3])),
        "SO2": abs(float(predicted_air_quality[0][4])),
        "CO": abs(float(predicted_air_quality[0][5])),
        "Ozone": abs(float(predicted_air_quality[0][6])),
	    "Aqi":""
	
    }
    
    concentrations = response
    sub_indices = {}
    for pollutant, conc in concentrations.items():
        if pollutant in breakpoints:
            sub_indices[pollutant] = calculate_aqi(conc, breakpoints[pollutant])

# Determine the final AQI (maximum of sub-indices)
    final_aqi = max(sub_indices.values())
    aqi_category = classify_aqi(final_aqi)
    response["Aqi"] = aqi_category
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
