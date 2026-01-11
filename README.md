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
  
## 2. Feature Pipeline

### STEP 1: 311 Feature Engineering
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

---

### STEP 2: Weather Feature Engineering
- **Weather Data Collection**:
  - Retrieve hourly weather data separately for each borough using Open-Meteo API.

- **Daily Aggregation**:
  - Aggregate hourly data to daily level:
    - `temperature_mean` (°C)
    - `precipitation_sum` (mm)
    - `wind_speed_mean` (m/s)

- **Date Alignment**:
  - Create a complete date grid with 5 rows per day (one per borough).

  
### STEP 3 : Feature Store:
write the two datasets into two different tables (feature groups) in Hopsworks.



## 3. Training Pipeline

### STEP 1: Feature Merging:
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


