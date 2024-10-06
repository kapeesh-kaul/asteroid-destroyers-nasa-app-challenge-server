'''
Available Routes:
### 1. `/get_top_planets` (POST)
   - **Input Parameters (JSON):**
     - `filepath` (string, optional): Path to the CSV file. Defaults to `'PSCompPars.csv'`.
     - `SNR0` (float, optional): Reference Signal-to-Noise Ratio. Default is `100`.
     - `D` (float, optional): Reference distance. Default is `6`.
     - `top_n` (integer, optional): Number of top planets to return. Default is `10`.
     - `SNR_filter` (integer, optional): SNR value to return count of values over threshold. Default is `5`.
     - `category` (string, optional): Category of planets to filter by (e.g., "Hot Jupiter", "Super Earth", etc.). Default is `None` (no filtering by category).
   - **Returns:** A JSON object containing:
     - `SNR_filter_count` (integer): The count of planets where the SNR is greater than the specified `SNR_filter`.
     - `top_planets` (list): A list of dictionaries representing the top `N` planets, each containing details such as:
       - `pl_name`: Planet name.
       - `hostname`: Host star name.
       - `sy_snum`: Number of stars in the system.
       - `disc_year`: Year of discovery.
       - `pl_rade`: Planet radius (Earth radii).
       - `st_rad`: Stellar radius (solar radii).
       - `st_teff`: Stellar effective temperature (K).
       - `sy_dist`: Distance to the system (pc).
       - `category`: Category of the planet.

### 2. `/get_nearest_neighbors` (POST)
   - **Input Parameters (JSON):**
     - `filepath` (string, optional): Path to the CSV file. Defaults to `'PSCompPars.csv'`.
     - `SNR0` (float, optional): Reference Signal-to-Noise Ratio. Default is `100`.
     - `D` (float, optional): Reference distance. Default is `6`.
     - `pl_name` (string, optional): Name of the planet to find neighbors for. Default is `'7 CMa b'`.
     - `k` (integer, optional): Number of nearest neighbors to return. Default is `5`.
     - `category` (string, optional): Category of planets to filter by (e.g., "Hot Jupiter", "Super Earth", etc.). Default is `None` (no filtering by category).
   - **Returns:** A JSON array containing dictionaries for each of the `K` nearest neighbors of the specified planet. Each dictionary includes:
     - `pl_name`: Planet name.
     - `hostname`: Host star name.
     - `sy_snum`: Number of stars in the system.
     - `disc_year`: Year of discovery.
     - `pl_rade`: Planet radius (Earth radii).
     - `st_rad`: Stellar radius (solar radii).
     - `st_teff`: Stellar effective temperature (K).
     - `sy_dist`: Distance to the system (pc).
     - `category`: Category of the planet.

### 3. `/get_planet_details` (POST)
   - **Input Parameters (JSON):**
     - `filepath` (string, optional): Path to the CSV file containing the exoplanet data. Defaults to `'PSCompPars.csv'`.
     - `pl_name` (string, optional): Name of the planet to get details for. Defaults to `'7 CMa b'`.
   - **Returns:** A JSON object containing:
     - `planet_details`: The full row of data for the specified planet as a dictionary.
     - `schema_description` (omitted): A dictionary providing descriptions of each column in the dataset. (Note: This is omitted in the current implementation based on the instructions).

   
Notes:
- Ensure the CSV file contains the necessary columns, including 'pl_name', 
  'hostname', 'ra', 'dec', 'sy_dist', 'st_rad', 'st_teff', 'pl_orbsmax', etc.
- The API handles missing values by filling numeric columns with their median 
  and non-numeric columns with 'unknown'.
'''
from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from flask_caching import Cache
import os

# Suppress all warnings
warnings.filterwarnings('ignore')

# Flask app
app = Flask(__name__)
CORS(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Global variables to keep track of the dataframe and file path
global_df = None
current_filepath = None

# The valid categories
VALID_CATEGORIES = ['Hot Jupiter', 'Super Earth', 'Earth-like', 'Ice Giant', 'Mini-Neptune']

# The ExoData class
class ExoData():
    def __init__(self, path):
        # Read the CSV file and perform initial transformations
        self.exoplanet_data = pd.read_csv(path, comment='#', delimiter=',')
        self.exoplanet_data = self.precompute_columns(self.exoplanet_data)

    def precompute_columns(self, df):
        # Validate schema
        self.validate_schema(df)
        
        # Extract the relevant columns
        columns_to_include = [
            'pl_name', 'hostname', 'ra', 'dec', 'sy_dist', 'st_rad', 'st_teff', 'pl_orbsmax', 'pl_orbeccen',
            'pl_orbincl', 'pl_rade', 'pl_eqt', 'pl_orbper', 'st_lum', 'sy_snum', 'disc_year'
        ]
        df = df[columns_to_include].copy()

        # Fill NaN values for numeric columns with their median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: pd.to_numeric(x, downcast='float').fillna(x.median()))

        # Fill NaN values for non-numeric columns with the placeholder 'unknown'
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna('unknown')

        # Convert RA and Dec from degrees to radians
        df['ra_rad'] = np.radians(df['ra'])
        df['dec_rad'] = np.radians(df['dec'])

        # Calculate Cartesian coordinates (X, Y, Z)
        df['X'] = df['sy_dist'] * np.cos(df['ra_rad']) * np.cos(df['dec_rad'])
        df['Y'] = df['sy_dist'] * np.sin(df['ra_rad']) * np.cos(df['dec_rad'])
        df['Z'] = df['sy_dist'] * np.sin(df['dec_rad'])

        # Calculate habitable zone boundaries
        df['habitable_zone_inner'] = np.sqrt(df['st_lum'] / 1.1)
        df['habitable_zone_outer'] = np.sqrt(df['st_lum'] / 0.53)

        # Habitability metrics
        df['habitable_zone_center_au'] = (df['habitable_zone_inner'] + df['habitable_zone_outer']) / 2
        df['habitable_zone_width_au'] = (df['habitable_zone_outer'] - df['habitable_zone_inner']) / 2
        df['hz_score'] = 1 - abs((df['pl_orbsmax'] - df['habitable_zone_center_au']) / df['habitable_zone_width_au'])
        df['hz_score'] = df['hz_score'].clip(lower=0)

        # Other scores
        df['size_score'] = np.exp(-((df['pl_rade'] - 1) ** 2) / 2)
        df['temp_score'] = np.exp(-((df['pl_eqt'] - 300) ** 2) / 2000)
        df['eccentricity_score'] = 1 - df['pl_orbeccen']
        df['eccentricity_score'] = df['eccentricity_score'].clip(lower=0)

        # Overall habitability score
        df['habitability_score'] = (df['hz_score'] *
                                    df['size_score'] *
                                    df['temp_score'] *
                                    df['eccentricity_score'])

        # Assign categories based on planet characteristics
        conditions = [
            (df['pl_rade'] > 10) & (df['pl_eqt'] > 1000),
            (df['pl_rade'].between(1.25, 2.0)) & (df['pl_eqt'].between(250, 500)),
            (df['pl_rade'] <= 1.25) & (df['pl_eqt'].between(250, 350)),
            (df['pl_rade'] > 10) & (df['pl_eqt'] < 250),
            (df['pl_rade'].between(2.0, 10)) & (df['pl_eqt'] < 1000)
        ]
        categories = ['Hot Jupiter', 'Super Earth', 'Earth-like', 'Ice Giant', 'Mini-Neptune']
        df['category'] = np.select(conditions, categories, default='Unknown')

        return df

    @staticmethod
    def validate_schema(df):
        required_columns = {'pl_name', 'hostname', 'ra', 'dec', 'sy_dist', 'st_rad', 'st_teff', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl', 'pl_rade', 'pl_eqt', 'pl_orbper', 'st_lum', 'sy_snum', 'disc_year'}
        if not required_columns.issubset(df.columns):
            raise ValueError("The provided file does not contain the required columns.")

# Helper function to load global DataFrame
def load_global_dataframe(filepath):
    global global_df, current_filepath
    if global_df is None or current_filepath != filepath:
        exo_data_instance = ExoData(filepath)
        global_df = exo_data_instance.precompute_columns(exo_data_instance.exoplanet_data)
        current_filepath = filepath

# @app.before_request
# def before_request_func():
filepath = 'PSCompPars.csv'
load_global_dataframe(filepath)

# Function to calculate SNR dynamically
def calculate_snr(df, SNR0, D):
    df['snr'] = SNR0 * ((df['st_rad'] * df['pl_rade'] * (D / 6)) /
                        ((df['sy_dist'] / 10) * df['pl_orbsmax'])) ** 2
    df['habitable'] = df['snr'] > 5
    return df

# Route to get top N planets by SNR
@app.route('/get_top_planets', methods=['POST'])
def get_top_planets():
    data = request.json
    SNR0 = data.get('SNR0', 100)
    D = data.get('D', 6)
    SNR_filter = data.get('SNR_filter', 5)
    top_n = data.get('top_n', 10)
    category_filter = data.get('category', None)

    # Create a temporary DataFrame to calculate SNR with user's parameters
    temp_df = global_df.copy()
    temp_df = calculate_snr(temp_df, SNR0, D)

    # Validate the category and apply the filter if it is valid
    if category_filter in VALID_CATEGORIES:
        temp_df = temp_df[temp_df['category'] == category_filter]

    # Get the top 'top_n' records based on the SNR column
    top_records = temp_df.nlargest(top_n, 'snr')

    # Select and rename the columns to match the desired JSON format
    top_records = top_records[['pl_name', 'hostname', 'sy_snum', 'disc_year', 'pl_rade', 'st_rad', 'st_teff', 'sy_dist', 'category']]
    
    # Get the count of rows where snr > SNR_filter
    SNR_filter_count = temp_df[temp_df['snr'] > SNR_filter].shape[0]

    result = {
        "SNR_filter_count": SNR_filter_count,
        "top_planets": top_records.to_dict(orient='records')
    }
    
    return jsonify(result)

# Route to get K nearest neighbors for a given planet name
@app.route('/get_nearest_neighbors', methods=['POST'])
def get_nearest_neighbors():
    data = request.json
    SNR0 = data.get('SNR0', 100)
    D = data.get('D', 6)
    pl_name = data.get('pl_name', '7 CMa b')
    k = data.get('k', 5)
    category_filter = data.get('category', None)

    # Create a temporary DataFrame to calculate SNR with user's parameters
    temp_df = global_df.copy()
    temp_df = calculate_snr(temp_df, SNR0, D)

    # Validate the category and apply the filter if it is valid
    if category_filter in VALID_CATEGORIES:
        temp_df = temp_df[temp_df['category'] == category_filter]

    # Select features for the KNN search
    features = ['X', 'Y', 'Z', 'st_rad', 'st_teff', 'pl_orbsmax', 'habitability_score']
    feature_data = temp_df[features]

    feature_data = feature_data.apply(lambda x: x.fillna(x.median()), axis=0)

    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)

    # Check if the given planet name exists
    if pl_name not in temp_df['pl_name'].values:
        return jsonify({"error": "Planet name not found"}), 404

    # Get the index of the planet with the given name
    target_index = temp_df[temp_df['pl_name'] == pl_name].index[0]
    target_features = feature_data_scaled[target_index].reshape(1, -1)

    # Perform KNN to find the nearest neighbors
    knn = NearestNeighbors(n_neighbors=k + 1)  # Include the target planet itself
    knn.fit(feature_data_scaled)
    distances, indices = knn.kneighbors(target_features)

    # Exclude the target planet itself from the results
    neighbor_indices = indices[0][1:]
    nearest_neighbors = temp_df.iloc[neighbor_indices]

    # Select the columns for the output
    nearest_neighbors = nearest_neighbors[['pl_name', 'hostname', 'sy_snum', 'disc_year', 'pl_rade', 'st_rad', 'st_teff', 'sy_dist', 'category']]

    result = nearest_neighbors.to_dict(orient='records')
    return jsonify(result)

# Route to get the full row for a given planet name
@app.route('/get_planet_details', methods=['POST'])
def get_planet_details():
    data = request.json
    pl_name = data.get('pl_name', '7 CMa b')

    # Check if the given planet name exists
    if pl_name not in global_df['pl_name'].values:
        return jsonify({"error": "Planet name not found"}), 404

    # Get the full row for the specified planet
    planet_details = global_df[global_df['pl_name'] == pl_name].to_dict(orient='records')[0]

    result = {
        "planet_details": planet_details
    }

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
