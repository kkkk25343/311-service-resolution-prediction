# 311-service-resolution-prediction
Project name: Dynamic Prediction of 311 Service Request Resolution Using Urban and Weather Data

## 1. Datasets
### 1.1 NYC 311 Service Requests Data
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
- **Time Range**: **2025-09-01 to 2025-12-17**
- **Geographic Coverage**: NYC five boroughs
  - Manhattan
  - Brooklyn
  - Queens
  - Bronx
  - Staten Island
- **Method**: One representative latitude/longitude point per borough is used to retrieve historical hourly weather data.
  
## 2. Feature Pipeline

### STEP 1: Time Feature Engineering
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

### STEP 2: Weather Feature Retrieval and Merging
- **Weather Data Collection**:
  - Retrieve hourly weather data separately for each borough using Open-Meteo API.

- **Daily Aggregation**:
  - Aggregate hourly data to daily level:
    - `temperature_mean` (°C)
    - `precipitation_sum` (mm)
    - `wind_speed_mean` (m/s)

- **Date Alignment**:
  - Create a complete date grid with 5 rows per day (one per borough).
  - Ensure full coverage from **2025-09-01 to 2025-12-17**.

- **Feature Merging**:
  - Merge weather features into `df_clean` using `(borough, date)` as keys.
  - Automatically select the timezone interpretation (local vs UTC→NY) that maximizes the matching rate.

- Output dataset: `df_final`, including:
  - `weather_temperature_mean`
  - `weather_precipitation_sum`
  - `weather_wind_speed_mean`

---

### STEP 3: Label Generation (Pre-training)
- **Resolution Time Calculation**:
  - `resolution_hours = (closed_date - created_date) / 3600`

- **Binary Label Construction**:
  - `label_48h = 1` if `resolution_hours ≤ 48`, otherwise `0`.

- **Data Filtering**:
  - Remove records with negative resolution time.

---

### Final Feature Assembly

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


