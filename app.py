from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import warnings

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

@app.route('/get_top_planets', methods=['POST'])
def get_top_planets():
    # Get parameters from the request
    data = request.json
    filepath = data.get('filepath')
    SNR0 = data.get('SNR0', 100)
    D = data.get('D', 6)
    top_n = data.get('top_n', 10)  # New parameter for the number of records to return

    # Create an instance of ExoData and transform the data
    exo_data_instance = ExoData(filepath, SNR0, D)
    transformed_data = exo_data_instance.transform()

    # Get the top 'top_n' records based on the SNR column
    top_records = transformed_data.nlargest(top_n, 'snr')

    # Select and rename the columns to match the desired JSON format
    top_records = top_records[[
        'pl_name', 'hostname', 'sy_snum', 'disc_year', 'pl_rade', 'st_rad', 'st_teff', 'sy_dist'
    ]]

    # Convert the top records to a JSON format
    result = top_records.to_dict(orient='records')

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
