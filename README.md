# Exoplanet Characterization and Visualization App
#### FrontEnd Repo: https://github.com/KlausMikhaelson/asteroid-destroyers-nasa-app-challenge-server

## Overview
This project provides a solution to map and characterize exoplanets, assisting the Habitable Worlds Observatory (HWO) in prioritizing potential planets for further study. The application integrates data processing, machine learning, and interactive visualization to facilitate the exploration of exoplanet data.

## Features
1. **Data Input & Processing**:
   - Uses `PSComp.csv` as the primary dataset for exoplanet information.
   - Data is processed with Pandas to fill missing values and convert radial parameters into 3D Cartesian coordinates for plotting.
   
2. **Machine Learning Models**:
   - **K-Nearest Neighbors (KNN)**: Finds exoplanets similar to a selected one based on their features.
   - **Rule-Based Classification**: Categorizes planets (e.g., Gas Giant, Super Earth) using size and temperature rules.
   - **LLM-Based Classification**: Utilizes the ChatGPT API to evaluate the habitability of planets by passing planetary parameters to a Large Language Model.

3. **Backend**:
   - Built using Flask to handle server requests, data processing, and ML model execution.
   - Implements caching for optimized data retrieval.

4. **Frontend**:
   - Created with React and 3.js for interactive visualizations of exoplanet data.
   - Displays exoplanets in a 3D space centered around Earth's planetary system.
   
5. **Visualization**:
   - Uses relative scaling to compare features like radius and distance.
   - Converts radial parameters (right ascension and declination) into 3D coordinates for accurate positioning.

## Usage
### Endpoints
1. **`/get_top_planets` (POST)**:
   - **Input**: Filepath, SNR0, D, top_n, SNR_filter, and category.
   - **Output**: Top N planets based on Signal-to-Noise Ratio (SNR), filtered by category if specified.

2. **`/get_nearest_neighbors` (POST)**:
   - **Input**: Filepath, SNR0, D, planet name, k (number of neighbors), and category.
   - **Output**: List of K nearest neighbor planets to the specified planet.

3. **`/get_planet_details` (POST)**:
   - **Input**: Filepath and planet name.
   - **Output**: Full data of the specified planet.

### Data Visualization
- Provides a 3D representation of exoplanets with Earth's planetary system as the reference.
- Focuses on planets observable from Earth-based telescopes.

## Technical Stack
- **Backend**: Flask, Pandas, Scikit-learn.
- **Frontend**: React, 3.js for interactive 3D visualizations.
- **Machine Learning**: KNN, rule-based classification, and LLM integration using ChatGPT API.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask server:
   ```bash
   python app.py
    ```
Open the React frontend to access the visualizations.
