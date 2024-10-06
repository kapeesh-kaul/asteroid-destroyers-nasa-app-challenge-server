'''
Available Routes:
1. /get_top_planets (POST)
   - Input Parameters (JSON):
     - filepath (string, optional): Path to the CSV file.
     - SNR0 (float, optional): Reference Signal-to-Noise Ratio. Default is 100.
     - D (float, optional): Reference distance. Default is 6.
     - top_n (integer, optional): Number of top planets to return. Default is 10.
     - SNR_filter (integer, optional): SNR value to return count of values over threshold
   - Returns a JSON structured like :
   {
        "SNR_filter_count": 22,
        "top_planets": [planets]
    }

2. /get_nearest_neighbors (POST)
   - Input Parameters (JSON):
     - filepath (string, optional): Path to the CSV file.
     - pl_name (string, optional): Name of the planet to find neighbors for.
     - k (integer, optional): Number of nearest neighbors to return. Default is 5.
   - Returns a JSON array of the K nearest neighbors for the specified planet.

3. /get_planet_details (POST)
   - Input Parameters (JSON):
     - filepath (string, optional): Path to the CSV file containing the exoplanet data.
     - pl_name (string, optional): Name of the planet to get details for. Defaults to '7 CMa b' if not provided.
   - Returns: A JSON object containing:
     - planet_details: The full row of data for the specified planet.
     - schema_description: A dictionary providing descriptions of each column in the dataset (Note: In this implementation, the schema description is omitted based on the current instructions).

   
Notes:
- Ensure the CSV file contains the necessary columns, including 'pl_name', 
  'hostname', 'ra', 'dec', 'sy_dist', 'st_rad', 'st_teff', 'pl_orbsmax', etc.
- The API handles missing values by filling numeric columns with their median 
  and non-numeric columns with 'unknown'.
'''

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
import os

# Suppress all warnings
warnings.filterwarnings('ignore')

# The ExoData class
class ExoData():
    def __init__(self, path, SNR0=100, D=6) -> None:
        self.path = path
        self.SNR0 = SNR0
        self.D = D

    def transform(self):
        exoplanet_data = pd.read_csv(self.path, comment='#', delimiter=',')

        # Extract the relevant columns
        columns_to_include = [
            'pl_name', 'hostname', 'ra', 'dec', 'sy_dist', 'st_rad', 'st_teff', 'pl_orbsmax', 'pl_orbeccen',
            'pl_orbincl', 'pl_rade', 'pl_eqt', 'pl_orbper', 'st_lum', 'sy_snum', 'disc_year'
        ]
        exoplanet_data = exoplanet_data[columns_to_include].copy()

        # Fill NaN values for numeric columns with their median
        numeric_cols = exoplanet_data.select_dtypes(include=[np.number]).columns
        exoplanet_data[numeric_cols] = exoplanet_data[numeric_cols].apply(lambda x: x.fillna(x.median()))

        # Fill NaN values for non-numeric columns with the placeholder 'unknown'
        non_numeric_cols = exoplanet_data.select_dtypes(exclude=[np.number]).columns
        exoplanet_data[non_numeric_cols] = exoplanet_data[non_numeric_cols].fillna('unknown')

        # Convert RA and Dec from degrees to radians
        exoplanet_data['ra_rad'] = np.radians(exoplanet_data['ra'])
        exoplanet_data['dec_rad'] = np.radians(exoplanet_data['dec'])

        # Calculate Cartesian coordinates (X, Y, Z)
        exoplanet_data['X'] = exoplanet_data['sy_dist'] * np.cos(exoplanet_data['ra_rad']) * np.cos(exoplanet_data['dec_rad'])
        exoplanet_data['Y'] = exoplanet_data['sy_dist'] * np.sin(exoplanet_data['ra_rad']) * np.cos(exoplanet_data['dec_rad'])
        exoplanet_data['Z'] = exoplanet_data['sy_dist'] * np.sin(exoplanet_data['dec_rad'])

        # Calculate habitable zone boundaries
        exoplanet_data['habitable_zone_inner'] = np.sqrt(exoplanet_data['st_lum'] / 1.1)
        exoplanet_data['habitable_zone_outer'] = np.sqrt(exoplanet_data['st_lum'] / 0.53)

        # Habitability metrics and SNR calculation
        exoplanet_data['habitable_zone_center_au'] = (exoplanet_data['habitable_zone_inner'] + exoplanet_data['habitable_zone_outer']) / 2
        exoplanet_data['habitable_zone_width_au'] = (exoplanet_data['habitable_zone_outer'] - exoplanet_data['habitable_zone_inner']) / 2
        exoplanet_data['hz_score'] = 1 - abs((exoplanet_data['pl_orbsmax'] - exoplanet_data['habitable_zone_center_au']) / exoplanet_data['habitable_zone_width_au'])
        exoplanet_data['hz_score'] = exoplanet_data['hz_score'].clip(lower=0)

        # Other scores
        exoplanet_data['size_score'] = np.exp(-((exoplanet_data['pl_rade'] - 1) ** 2) / 2)
        exoplanet_data['temp_score'] = np.exp(-((exoplanet_data['pl_eqt'] - 300) ** 2) / 2000)
        exoplanet_data['eccentricity_score'] = 1 - exoplanet_data['pl_orbeccen']
        exoplanet_data['eccentricity_score'] = exoplanet_data['eccentricity_score'].clip(lower=0)

        # Overall habitability score
        exoplanet_data['habitability_score'] = (exoplanet_data['hz_score'] *
                                                exoplanet_data['size_score'] *
                                                exoplanet_data['temp_score'] *
                                                exoplanet_data['eccentricity_score'])

        # SNR calculation
        exoplanet_data['snr'] = self.SNR0 * ((exoplanet_data['st_rad'] * exoplanet_data['pl_rade'] * (self.D / 6)) /
                                             ((exoplanet_data['sy_dist'] / 10) * exoplanet_data['pl_orbsmax'])) ** 2
        exoplanet_data['habitable'] = exoplanet_data['snr'] > 5

        return exoplanet_data

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_server_status():
    return jsonify({"message": "WE ONON"}), 200

# Route to get top N planets by SNR
@app.route('/get_top_planets', methods=['POST'])
def get_top_planets():
    data = request.json
    filepath = data.get('filepath', 'PSCompPars.csv')
    SNR0 = data.get('SNR0', 100)
    D = data.get('D', 6)
    SNR_filter = data.get('SNR_filter', 5)
    top_n = data.get('top_n', 10)  # Default to 10 if not specified

    # Create an instance of ExoData and transform the data
    exo_data_instance = ExoData(filepath, SNR0, D)
    transformed_data = exo_data_instance.transform()

    # Get the top 'top_n' records based on the SNR column
    top_records = transformed_data.nlargest(top_n, 'snr')

    # Select and rename the columns to match the desired JSON format
    top_records = top_records[[
        'pl_name', 'hostname', 'sy_snum', 'disc_year', 'pl_rade', 'st_rad', 'st_teff', 'sy_dist'
    ]]
    # Get the count of rows where snr > 5
    SNR_filter_count = transformed_data[transformed_data['snr'] > SNR_filter].shape[0]

    # Convert the top records to a JSON format
    result = {
        "SNR_filter_count": SNR_filter_count,
        "top_planets": top_records.to_dict(orient='records')
    }
    
    return jsonify(result)

# Route to get K nearest neighbors for a given planet name
@app.route('/get_nearest_neighbors', methods=['POST'])
def get_nearest_neighbors():
    data = request.json
    filepath = data.get('filepath', 'PSCompPars.csv')
    pl_name = data.get('pl_name', '7 CMa b')
    k = data.get('k', 5)  # Default to 5 neighbors if not specified

    # Create an instance of ExoData and transform the data
    exo_data_instance = ExoData(filepath)
    transformed_data = exo_data_instance.transform()

    # Select features for the KNN search
    features = ['X', 'Y', 'Z', 'st_rad', 'st_teff', 'pl_orbsmax', 'habitability_score']
    feature_data = transformed_data[features]

    # Fill any remaining NaN values in the features with their median
    feature_data = feature_data.apply(lambda x: x.fillna(x.median()), axis=0)

    # Check if the given planet name exists
    if pl_name not in transformed_data['pl_name'].values:
        return jsonify({"error": "Planet name not found"}), 404

    # Get the index of the planet with the given name
    target_index = transformed_data[transformed_data['pl_name'] == pl_name].index[0]
    target_features = feature_data.loc[target_index].values.reshape(1, -1)

    # Perform KNN to find the nearest neighbors
    knn = NearestNeighbors(n_neighbors=k + 1)  # Include the target planet itself
    knn.fit(feature_data)
    distances, indices = knn.kneighbors(target_features)

    # Exclude the target planet itself from the results
    neighbor_indices = indices[0][1:]
    nearest_neighbors = transformed_data.iloc[neighbor_indices]

    # Select the columns for the output
    nearest_neighbors = nearest_neighbors[[
        'pl_name', 'hostname', 'sy_snum', 'disc_year', 'pl_rade', 'st_rad', 'st_teff', 'sy_dist'
    ]]

    # Convert to JSON format
    result = nearest_neighbors.to_dict(orient='records')

    return jsonify(result)

# Define the full schema description dictionary

# Route to get the full row for a given planet name along with the schema description
@app.route('/get_planet_details', methods=['POST'])
def get_planet_details():
    data = request.json
    filepath = data.get('filepath', 'PSCompPars.csv')
    pl_name = data.get('pl_name', '7 CMa b')  # Default to '7 CMa b' if not provided

    # Create an instance of ExoData and transform the data
    exo_data_instance = ExoData(filepath)
    transformed_data = exo_data_instance.transform()

    # Check if the given planet name exists
    if pl_name not in transformed_data['pl_name'].values:
        return jsonify({"error": "Planet name not found"}), 404

    # Get the full row for the specified planet
    planet_details = transformed_data[transformed_data['pl_name'] == pl_name].to_dict(orient='records')[0]

    # Prepare the final response with the predefined schema description
    result = {
        "planet_details": planet_details
    }

    return jsonify(result)



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
