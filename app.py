from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from zipfile import ZipFile
import os

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Specify the path to the zip file and extraction path
zip_file_path = "indonesian_tourism_dataset.zip"
extract_path = "dataset"

# Extract the zip file
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List the extracted files
extracted_files = os.listdir(extract_path)

# Load datasets
rating = pd.read_csv(os.path.join(extract_path, 'tourism_rating.csv'))
place = pd.read_csv(os.path.join(extract_path, 'tourism_with_id.csv'))
user = pd.read_csv(os.path.join(extract_path, 'user.csv'))

# Merge data based on 'Place_Id'
data = pd.merge(rating, place, on='Place_Id')
data = pd.merge(data, user, on='User_Id')

# Drop unnecessary columns
data.drop(['Time_Minutes', 'Coordinate', 'Location', 'Unnamed: 11', 'Unnamed: 12'], axis=1, inplace=True)

# Initialize label encoder
label_encoder = LabelEncoder()
label_encoder.fit(data['Category'])

def recommend(age, category, lat, long):
    category_encoded = label_encoder.transform([category])[0]

    # Create input arrays for each place
    places = data['Place_Id'].unique()
    age_inputs = np.full(len(places), age)
    category_inputs = np.full(len(places), category_encoded)
    lat_inputs = np.full(len(places), lat)
    long_inputs = np.full(len(places), long)

    # Use the model to make predictions for each place
    predictions = model.predict([places, age_inputs, category_inputs, lat_inputs, long_inputs]).flatten()

    # Get the indices of places with the highest ratings
    top_indices = np.argsort(predictions)[::-1][:20]

    # Return recommended places (filter out duplicates)
    recommended_places = data.loc[data['Place_Id'].isin(places[top_indices])]
    unique_recommendations = recommended_places.drop_duplicates(subset=['Place_Id'])

    return unique_recommendations


@app.route("/predict")
def home():
    return "Welcome to API!"

@app.route('/predict', methods=['POST'])
def predict():
    age = request.json['age']
    category = request.json['category']
    lat = request.json['lat']
    long = request.json['long']
    recommendations = recommend(age, category, lat, long)
    result = recommendations[['Place_Name', 'Rating', 'Lat', 'Long']]
    return result.to_json(orient='records')

if __name__ == "__main__":
    app.run (host='0.0.0.0', port=int(os.environ.get("PORT", 8080)),debug=True)