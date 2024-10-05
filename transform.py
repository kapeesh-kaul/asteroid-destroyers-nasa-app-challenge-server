"""
# Exoplanet Data Definition (19 Columns)

1. star_ra_deg: Right Ascension of the star in degrees.
2. star_dec_deg: Declination of the star in degrees.
3. star_distance_pc: Distance to the star in parsecs.
4. star_radius_solar: Radius of the star in solar radii.
5. star_teff_k: Effective temperature of the star in Kelvin.
6. planet_orbit_semi_major_axis_au: Semi-major axis of the planet's orbit in astronomical units (AU), representing the average distance from the star.
7. planet_orbit_eccentricity: Eccentricity of the planet's orbit, indicating how elongated the orbit is. A value of 0 represents a circular orbit.
8. planet_orbit_inclination_deg: Inclination of the planet's orbit in degrees, relative to the line of sight.
9. planet_radius_earth: Radius of the planet in Earth radii.
10. planet_equilibrium_temp_k: Equilibrium temperature of the planet in Kelvin, estimating the surface temperature based on its distance from the star and stellar luminosity.
11. planet_orbital_period_days: Orbital period of the planet in days, representing the time it takes for the planet to complete one orbit around its star.
12. star_luminosity_solar: Luminosity of the star in solar units, indicating the total energy output of the star.
13. ra_rad: Right Ascension converted to radians for Cartesian coordinate calculations.
14. dec_rad: Declination converted to radians for Cartesian coordinate calculations.
15. star_cartesian_x: Cartesian X-coordinate of the star based on RA, Dec, and distance, used for 3D spatial visualization.
16. star_cartesian_y: Cartesian Y-coordinate of the star based on RA, Dec, and distance, used for 3D spatial visualization.
17. star_cartesian_z: Cartesian Z-coordinate of the star based on RA, Dec, and distance, used for 3D spatial visualization.
18. star_habitable_zone_inner_au: Inner boundary of the star's habitable zone in astronomical units (AU), calculated using the star's luminosity.
19. star_habitable_zone_outer_au: Outer boundary of the star's habitable zone in astronomical units (AU), calculated using the star's luminosity.
20. habitability_score:

"""

import numpy as np
import pandas as pd

class ExoData():
    def __init__(self, path, SNR0 = 100, D = 6) -> None:
        self.path = path
        self.SNR0 = SNR0
        self.D = D

    
    def transform(self):        
        exoplanet_data = pd.read_csv(self.path, comment='#', delimiter=',')

        # Extract the relevant columns from the main DataFrame for calculations
        columns_to_include = [
            'pl_name', 'hostname', 'ra', 'dec', 'sy_dist', 'st_rad', 'st_teff', 'pl_orbsmax', 'pl_orbeccen', 
            'pl_orbincl', 'pl_rade', 'pl_eqt', 'pl_orbper', 'st_lum'
        ]
        exoplanet_data = exoplanet_data[columns_to_include].copy()

        # Convert RA and Dec from degrees to radians for Cartesian coordinate calculations
        exoplanet_data['ra_rad'] = np.radians(exoplanet_data['ra'])
        exoplanet_data['dec_rad'] = np.radians(exoplanet_data['dec'])

        # Calculate Cartesian coordinates (X, Y, Z) using distance (in parsecs), RA, and Dec
        exoplanet_data['X'] = exoplanet_data['sy_dist'] * np.cos(exoplanet_data['ra_rad']) * np.cos(exoplanet_data['dec_rad'])
        exoplanet_data['Y'] = exoplanet_data['sy_dist'] * np.sin(exoplanet_data['ra_rad']) * np.cos(exoplanet_data['dec_rad'])
        exoplanet_data['Z'] = exoplanet_data['sy_dist'] * np.sin(exoplanet_data['dec_rad'])

        # Calculate the inner and outer boundaries of the habitable zone using stellar luminosity (st_lum)
        # Inner boundary: sqrt(Luminosity / 1.1)
        # Outer boundary: sqrt(Luminosity / 0.53)
        exoplanet_data['habitable_zone_inner'] = np.sqrt(exoplanet_data['st_lum'] / 1.1)
        exoplanet_data['habitable_zone_outer'] = np.sqrt(exoplanet_data['st_lum'] / 0.53)

        exoplanet_data.rename(columns={
        'ra': 'star_ra_deg',
        'dec': 'star_dec_deg',
        'sy_dist': 'star_distance_pc',
        'st_rad': 'star_radius_solar',
        'st_teff': 'star_teff_k',
        'pl_orbsmax': 'planet_orbit_semi_major_axis_au',
        'pl_orbeccen': 'planet_orbit_eccentricity',
        'pl_orbincl': 'planet_orbit_inclination_deg',
        'pl_rade': 'planet_radius_earth',
        'pl_eqt': 'planet_equilibrium_temp_k',
        'pl_orbper': 'planet_orbital_period_days',
        'st_lum': 'star_luminosity_solar',
        'X': 'star_cartesian_x',
        'Y': 'star_cartesian_y',
        'Z': 'star_cartesian_z',
        'habitable_zone_inner': 'star_habitable_zone_inner_au',
        'habitable_zone_outer': 'star_habitable_zone_outer_au'
        }, inplace=True)

        # Calculate the habitability metrics for each planet

        # Helper columns for habitable zone center and width
        exoplanet_data['habitable_zone_center_au'] = (exoplanet_data['star_habitable_zone_inner_au'] + exoplanet_data['star_habitable_zone_outer_au']) / 2
        exoplanet_data['habitable_zone_width_au'] = (exoplanet_data['star_habitable_zone_outer_au'] - exoplanet_data['star_habitable_zone_inner_au']) / 2

        # 1. Distance from the Habitable Zone (HZ Score)
        exoplanet_data['hz_score'] = 1 - abs((exoplanet_data['planet_orbit_semi_major_axis_au'] - exoplanet_data['habitable_zone_center_au']) / exoplanet_data['habitable_zone_width_au'])
        exoplanet_data['hz_score'] = exoplanet_data['hz_score'].clip(lower=0)  # Ensure scores are not negative

        # 2. Planet Size (Size Score) - Gaussian centered at 1 Earth radius
        exoplanet_data['size_score'] = np.exp(-((exoplanet_data['planet_radius_earth'] - 1) ** 2) / 2)

        # 3. Equilibrium Temperature (Temp Score) - Gaussian centered at 300 K
        exoplanet_data['temp_score'] = np.exp(-((exoplanet_data['planet_equilibrium_temp_k'] - 300) ** 2) / 2000)

        # 4. Orbital Eccentricity (Eccentricity Score) - Score decreases as eccentricity increases
        exoplanet_data['eccentricity_score'] = 1 - exoplanet_data['planet_orbit_eccentricity']
        exoplanet_data['eccentricity_score'] = exoplanet_data['eccentricity_score'].clip(lower=0)  # Ensure scores are not negative

        # 5. Calculate the overall Habitability Score
        exoplanet_data['habitability_score'] = (exoplanet_data['hz_score'] *
                                                exoplanet_data['size_score'] *
                                                exoplanet_data['temp_score'] *
                                                exoplanet_data['eccentricity_score'])
             
        exoplanet_data['snr'] = self.SNR0 * ((exoplanet_data['star_radius_solar'] * exoplanet_data['planet_radius_earth'] * (self.D / 6)) / ((exoplanet_data['star_distance_pc'] / 10) * exoplanet_data['planet_orbit_semi_major_axis_au'])) ** 2
        
        return exoplanet_data

if __name__ == "__main__":
    ExoData('PSCompPars_2024.10.05_08.34.19.csv').transform().to_csv('exoplanet_visualization.csv')