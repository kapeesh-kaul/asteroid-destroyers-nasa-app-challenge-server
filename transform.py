"""
# Exoplanet Data Definition

1. star_ra_deg: Right Ascension of the star in degrees.
2. star_dec_deg: Declination of the star in degrees.
3. star_distance_pc: Distance to the star in parsecs.
4. star_radius_solar: Radius of the star in solar radii.
5. star_teff_k: Effective temperature of the star in Kelvin.
6. planet_orbit_semi_major_axis_au: Semi-major axis of the planet's orbit in astronomical units (AU).
7. planet_orbit_eccentricity: Eccentricity of the planet's orbit, indicating how elongated the orbit is.
8. planet_orbit_inclination_deg: Inclination of the planet's orbit in degrees.
9. planet_radius_earth: Radius of the planet in Earth radii.
10. planet_equilibrium_temp_k: Equilibrium temperature of the planet in Kelvin.
11. planet_orbital_period_days: Orbital period of the planet in days.
12. star_luminosity_solar: Luminosity of the star in solar units.
13. ra_rad: Right Ascension converted to radians for Cartesian coordinate calculations.
14. dec_rad: Declination converted to radians for Cartesian coordinate calculations.
15. star_cartesian_x: Cartesian X-coordinate of the star based on RA, Dec, and distance.
16. star_cartesian_y: Cartesian Y-coordinate of the star based on RA, Dec, and distance.
17. star_cartesian_z: Cartesian Z-coordinate of the star based on RA, Dec, and distance.
18. star_habitable_zone_inner_au: Inner boundary of the star's habitable zone in astronomical units (AU).
19. star_habitable_zone_outer_au: Outer boundary of the star's habitable zone in astronomical units (AU).
"""

import numpy as np
import pandas as pd

class ExoData():
    def __init__(self, path) -> None:
        self.path = path

    def transform(self):        
        exoplanet_data = pd.read_csv(self.path, comment='#', delimiter=',')

        # Extract the relevant columns from the main DataFrame for calculations
        columns_to_include = [
            'ra', 'dec', 'sy_dist', 'st_rad', 'st_teff', 'pl_orbsmax', 'pl_orbeccen', 
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

        return exoplanet_data

if __name__ == "__main__":
    ExoData('PSCompPars_2024.10.05_08.34.19.csv').transform().to_csv('exoplanet_visualization.csv')