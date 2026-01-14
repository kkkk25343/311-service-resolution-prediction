# 311-service-resolution-prediction
Project name: Dynamic Prediction of 311 Service Request Resolution Using Urban and Weather Data

Our project focuses on predicting the resolution efficiency of NYC 311 service requests, which are non-emergency reports submitted by city residents. By combining urban service data with environmental factors to better understand what influences how quickly issues are addressed. Using the publicly available NYC 311 dataset from the city’s open data portal and historical weather data from the Open-Meteo API, we construct a rich feature set that incorporates temporal, categorical, and weather-related information.

We frame resolution prediction as both a classification task (whether a request will be resolved within 48 hours) and a regression task (estimating the total time to resolution), training models such as XGBoost to produce actionable insights. By integrating multiple data sources and advanced feature engineering, our work aims to provide data-driven predictions that support operational planning and transparency for urban service delivery.

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

The notebook `2_311_services_data_updates.ipynb` is responsible for retrieving newly available NYC 311 service requests and weather data from their respective online sources. It applies the same data cleaning, feature engineering, and transformation logic as used during initial dataset construction. It writes the updated features into the corresponding feature groups in Hopsworks. Users can setup their wanted time range to update the feature store.


## 3. Training Pipeline

### STEP 1: Training Data Construction
  - Read the NYC 311 request features and weather features from two feature groups in Hopsworks.
  - Join the two feature groups using `(borough, date)` as the primary keys to construct the training dataset.
  - Select the timezone interpretation (local time vs. UTC→NY conversion) that maximizes the feature matching rate.

- Output dataset: `df_merged`, including:
  - `weather_temperature_mean`
  - `weather_precipitation_sum`
  - `weather_wind_speed_mean`

---

### STEP 2: Label Generation (Pre-training)
Construct the prediction target by transforming raw timestamp information into a supervised learning label that reflects service resolution efficiency.

- **Resolution Time Calculation**:
  - Compute the service resolution duration in hours as:
    - `resolution_hours = (closed_date - created_date) / 3600`
  - This value represents the total time required to resolve a 311 service request.

- **Binary Label Construction**:
  - Define a binary classification label `label_48h` to indicate whether a service request is resolved within a practical time threshold:
    - `label_48h = 1` if `resolution_hours ≤ 48`
    - `label_48h = 0` otherwise
  - The 48-hour threshold is chosen to reflect a commonly used operational service-level expectation and to balance label interpretability with class distribution.

- **Data Filtering and Quality Control**:
  - Remove records with negative or invalid resolution times, which may arise from data inconsistencies or timestamp errors.
  - Exclude unresolved or ongoing service requests that lack a valid `closed_date`, ensuring that all labels are well-defined.

As a result, this part produces a clean and interpretable target variable suitable for supervised classification models.


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
 
### STEP 4: Model Training and Validation

After constructing the final training dataset, we train supervised learning models to address two complementary prediction tasks: a binary classification task and a regression task.

#### 4.1 Temporal Train–Validation Split

To reflect real-world deployment conditions and avoid temporal leakage, the dataset is split chronologically:

- **Training Set**: Service requests created between **2025-09-01** and **2025-12-01**
- **Validation Set**: All service requests created after **2025-12-01**

This time-based split ensures that models are trained on historical data and evaluated on future, unseen requests.

---

#### 4.2 Classification Model: Resolution Within 48 Hours

- **Task**: Binary classification  
  Predict whether a NYC 311 service request will be resolved within 48 hours.
- **Target Label**: `label_48h`
- **Model**: XGBoost Classifier
- **Input Features**:  
  - Categorical features (e.g., agency, complaint type, borough)  
  - Temporal features (e.g., hour, weekday, workday indicators)  
  - Weather features (temperature, precipitation, wind speed)

The classification model outputs the probability that a service request will be resolved within the 48-hour threshold, providing an interpretable and actionable prediction aligned with service-level expectations. 
<img width="618" height="262" alt="image" src="https://github.com/user-attachments/assets/44b6a2c4-e8c9-4166-8b13-7024510d52eb" />

---

#### 4.3 Regression Model: Resolution Time Prediction

- **Task**: Regression  
  Predict the total resolution time (in hours) for a service request.
- **Target Label**: `resolution_hours`
- **Model**: XGBoost Regressor
- **Input Features**:  
  The same feature set used in the classification task is reused to enable a consistent comparison between modeling objectives.

The regression model provides a continuous estimate of expected resolution time, offering finer-grained insight into service performance variability.
<img width="1360" height="548" alt="image" src="https://github.com/user-attachments/assets/8fb89c30-3184-405a-9297-2123c881a773" />



---

#### 4.4 Model Registry and Versioning

Both trained models are registered in the Hopsworks Model Registry for versioned storage and reproducibility:

- **Classification Model**: `nyc_311_within48h_xgb`
- **Regression Model**: `nyc_311_resolution_hours_xgb`

Each model is stored with associated metadata, including training data version, feature schema, and model parameters, enabling traceability and future deployment.
<img width="2352" height="286" alt="image" src="https://github.com/user-attachments/assets/a7d1074a-19e9-4e95-8910-d8c2ce8e27cd" />

---

## 4. Batch Inference Pipeline

To demonstrate the end-to-end usability of the trained models, we implement a batch inference pipeline with a lightweight interactive user interface built using Streamlit.

### 4.1 Input Data Selection

The batch inference pipeline retrieves the most recent NYC 311 service requests from the feature store.  
Users can specify the number of latest records to process (e.g., the most recent 100 requests), allowing flexible control over batch size.

The pipeline automatically:
- Reads the latest available 311 request features from Hopsworks
- Applies the same feature schema used during model training
- Ensures consistency between training and inference data

---

### 4.2 Batch Prediction

For each selected batch, the pipeline performs inference using two registered XGBoost models:

- **Classification Model** (`nyc_311_within48h_xgb`):  
  Predicts the probability that a service request will be resolved within 48 hours.

- **Regression Model** (`nyc_311_resolution_hours_xgb`):  
  Predicts the expected resolution time (in hours).

Both models are loaded from the Hopsworks Model Registry, ensuring versioned and reproducible inference.

---

### 4.3 Output and Visualization

The inference results are presented through an interactive Streamlit interface, including:

- **Summary statistics**, such as:
  - Average probability of resolution within 48 hours
  - Proportion of requests predicted to be resolved within 48 hours
  - Average predicted resolution time

- **Tabular predictions**, showing for each request:
  - Key request attributes (e.g., agency, complaint type, borough)
  - Predicted probability and binary outcome for the 48-hour resolution task
  - Predicted resolution time in hours

This interface allows users to easily explore and interpret batch prediction results.

---

### 4.4 Execution

The batch inference application can be launched locally using:

```bash
streamlit run app.py
```

<img width="3072" height="1582" alt="image" src="https://github.com/user-attachments/assets/d5d6bfe7-a3b5-4117-88d0-c310390ed8d3" />

---


