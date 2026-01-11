# 311-service-resolution-prediction
Project name: Dynamic Prediction of 311 Service Request Resolution Using Urban and Weather Data

## 1. Datasets
### 1.1 NYC 311 Service Requests Data
NYC 311 Service Requests is a continuously updated, city-wide dataset provided via the NYC Open Data Socrata API, containing detailed records of non-emergency service requests submitted by residents, including request type, location, timestamps, and resolution information.
- **Source**: NYC Open Data (Socrata API)  
  https://data.cityofnewyork.us/resource/erm2-nwe9.json
- **Time Range**: Requests created after **2025-09-01**
- **Selected Fields**:
  - `created_date`
  - `closed_date`
  - `agency`
  - `agency_name`
  - `complaint_type`
  - `descriptor`
  - `location_type`
  - `borough`
### 1.2 Weather Data
- **Source**: Open-Meteo Archive API  
  https://archive-api.open-meteo.com/v1/archive
- **Time Range**: Weather Data collected after **2025-09-01**
- **Geographic Coverage**: NYC five boroughs
  - Manhattan
  - Brooklyn
  - Queens
  - Bronx
  - Staten Island
- **Method**: One representative latitude/longitude point per borough is used to retrieve historical hourly weather data. For each of the five districts, we select one station from a weather website to represent the overall weather conditions of that district.
```python
BOROUGH_COORDS = {
    "MANHATTAN": (40.7829, -73.9654),
    "BROOKLYN": (40.6928, -73.9903),
    "QUEENS": (40.7769, -73.8740),
    "BRONX": (40.8506, -73.8769),
    "STATEN ISLAND": (40.6437, -74.0736),
}
```
## 2. Feature Backfill Pipeline

### STEP 1: Feature Engineering for NYC 311 Service Requests
Perform offline feature engineering on NYC 311 service request data by extracting time-aware and calendar-based features that capture temporal patterns relevant to service resolution behavior.
- **Timezone Conversion**:
  - Convert `created_date` from UTC to New York local time (`America/New_York`).

- **Extract Temporal Features**:
  - `date`: calendar date (used for weather merging)
  - `hour`: hour of day (0–23)
  - `weekday`: day of week (0 = Monday, 6 = Sunday)

- **Holiday Identification**:
  - Use `holidays.US()` to determine U.S. federal holidays.
  - Generate binary feature `is_holiday`.

- **Workday Feature**:
  - `is_work_day = 1` if Monday–Friday and not a holiday; otherwise `0`.

- **Working Hours Feature**:
  - `is_work_hours = 1` if workday and time between 09:00–17:00; otherwise `0`.

- **Cleanup**:
  - Drop intermediate columns:
    - `created_date_ny`, `date`, `hour`, `weekday`, `is_holiday`.
   
The figure below presents an overview of NYC 311 service requests used in this project.
<img width="1594" height="790" alt="image" src="https://github.com/user-attachments/assets/90a0157b-d512-44c9-90f4-53fe69a3e814" />

---

### STEP 2: Weather Feature Engineering
Construct weather-based features by retrieving hourly meteorological data.
- **Weather Data Collection**:
  - Retrieve hourly weather data separately for each borough using Open-Meteo API.

- **Daily Aggregation**:
  - Compute daily weather features by averaging hourly observations within each day, with precipitation aggregated as daily totals.
    - `temperature_mean` (°C)
    - `precipitation_sum` (mm)
    - `wind_speed_mean` (m/s)

- **Date Alignment**:
  - Create a complete date grid with 5 rows per day (one per borough).
The figure below presents the weather data with geography information used in this project.
<img width="1204" height="540" alt="image" src="https://github.com/user-attachments/assets/97492d58-430c-4ec1-887b-f825c87f9c5b" />

  
### STEP 3 : Feature Store:
Insert two datasets into two feature groups in Hopsworks.
<img width="2014" height="278" alt="image" src="https://github.com/user-attachments/assets/0e5935e0-fc96-4bd6-acf4-b58e63253e60" />



## 3. Training Pipeline

### STEP 1: 
  - write the two datasets into two different tables (feature groups) in Hopsworks.
  - Merge weather features into `df_clean`(a dataframe) using `(borough, date)` as keys.
  - Automatically select the timezone interpretation (local vs UTC→NY) that maximizes the matching rate.

- Output dataset: `df_final`, including:
  - `weather_temperature_mean`
  - `weather_precipitation_sum`
  - `weather_wind_speed_mean`

---

### STEP 2: Label Generation (Pre-training)
- **Resolution Time Calculation**:
  - `resolution_hours = (closed_date - created_date) / 3600`

- **Binary Label Construction**:
  - `label_48h = 1` if `resolution_hours ≤ 48`, otherwise `0`.

- **Data Filtering**:
  - Remove records with negative resolution time.

---

### STEP 3: Final Feature Assembly

#### Categorical Features (6)
- `agency`
- `agency_name`
- `complaint_type`
- `descriptor`
- `location_type`
- `borough`

#### Numerical Features (9)
- **Temporal Features**:
  - `is_work_day`
  - `is_work_hours`
  - `created_hour`
  - `created_weekday`
  - `created_month`
  - `is_weekend`
- **Weather Features**:
  - `weather_temperature_mean`
  - `weather_precipitation_sum`
  - `weather_wind_speed_mean`

---


