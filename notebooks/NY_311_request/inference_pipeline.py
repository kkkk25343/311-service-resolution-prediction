# inference_pipeline.py
import os
import json
import requests
import pandas as pd
import pytz
import holidays
import joblib

NY_TZ = pytz.timezone("America/New_York")
BASE_311 = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# 你说 key 写死，而且 UI 不出现，那 Socrata token 也写死在这里（没有就空）
SOCRATA_APP_TOKEN = ""

CLASSIFIER_MODEL_NAME = "nyc_311_within48h_xgb"
REGRESSOR_MODEL_NAME = "nyc_311_resolution_hours_xgb"

MODEL_VERSION = None  # 都用 latest


BOROUGH_COORDS = {
    "MANHATTAN": (40.7829, -73.9654),
    "BROOKLYN": (40.6928, -73.9903),
    "QUEENS": (40.7769, -73.8740),
    "BRONX": (40.8506, -73.8769),
    "STATEN ISLAND": (40.6437, -74.0736),
}


def fetch_latest_311(limit: int = 100) -> pd.DataFrame:
    headers = {}
    if SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = SOCRATA_APP_TOKEN

    params = {"$limit": int(limit), "$order": "created_date DESC"}
    r = requests.get(BASE_311, params=params, headers=headers, timeout=120)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def fetch_weather_daily_for_dates(dates: list[str]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame(columns=[
            "borough", "service_date",
            "weather_temperature_mean",
            "weather_precipitation_sum",
            "weather_wind_speed_mean",
        ])

    unique_dates = sorted(set(dates))
    start_date, end_date = unique_dates[0], unique_dates[-1]

    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_vars = ["temperature_2m", "precipitation", "wind_speed_10m"]

    frames = []
    for borough, (lat, lon) in BOROUGH_COORDS.items():
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": "America/New_York",
        }
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        js = r.json()
        hourly = (js or {}).get("hourly") or {}
        if "time" not in hourly:
            continue

        dfh = pd.DataFrame(hourly)
        dfh["datetime"] = pd.to_datetime(dfh["time"], errors="coerce")
        dfh["service_date"] = dfh["datetime"].dt.date.astype(str)
        dfh["borough"] = borough

        daily = (
            dfh.groupby(["borough", "service_date"], as_index=False)
            .agg(
                weather_temperature_mean=("temperature_2m", "mean"),
                weather_precipitation_sum=("precipitation", "sum"),
                weather_wind_speed_mean=("wind_speed_10m", "mean"),
            )
        )
        frames.append(daily)

    if not frames:
        return pd.DataFrame(columns=[
            "borough", "service_date",
            "weather_temperature_mean",
            "weather_precipitation_sum",
            "weather_wind_speed_mean",
        ])

    weather = pd.concat(frames, ignore_index=True)

    grid = pd.MultiIndex.from_product(
        [list(BOROUGH_COORDS.keys()), unique_dates],
        names=["borough", "service_date"]
    ).to_frame(index=False)

    return grid.merge(weather, on=["borough", "service_date"], how="left")


def preprocess_311_like_training(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw

    keep = [
        "created_date",
        "agency",
        "agency_name",
        "complaint_type",
        "descriptor",
        "location_type",
        "borough",
    ]
    keep = [c for c in keep if c in df_raw.columns]
    df = df_raw[keep].copy()

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce", utc=True)
    created_ny = df["created_date"].dt.tz_convert(NY_TZ)

    df["created_hour"] = created_ny.dt.hour
    df["created_weekday"] = created_ny.dt.weekday
    df["created_month"] = created_ny.dt.month
    df["is_weekend"] = (df["created_weekday"] >= 5).astype(int)

    us_holidays = holidays.US()
    local_date = created_ny.dt.date
    df["is_holiday"] = local_date.apply(lambda x: 1 if x in us_holidays else 0)
    df["is_work_day"] = ((df["created_weekday"] < 5) & (df["is_holiday"] == 0)).astype(int)
    df["is_work_hours"] = ((df["is_work_day"] == 1) & (df["created_hour"] >= 9) & (df["created_hour"] < 17)).astype(int)
    df = df.drop(columns=["is_holiday"])

    df["service_date"] = created_ny.dt.date.astype(str)
    df["borough"] = df["borough"].astype(str).str.upper().str.strip()

    return df


def join_weather(df_311: pd.DataFrame, weather_daily: pd.DataFrame) -> pd.DataFrame:
    if df_311.empty:
        return df_311
    w = weather_daily.copy()
    w["borough"] = w["borough"].astype(str).str.upper().str.strip()
    return df_311.merge(w, on=["borough", "service_date"], how="left")


# def load_model_from_registry(mr, model_name: str = MODEL_NAME, version: int | None = MODEL_VERSION):
#     """
#     mr: project.get_model_registry() 得到的对象，由 app.py 传进来
#     """
#     if version is None:
#         models = mr.get_models(model_name)
#         if not models:
#             raise RuntimeError(f"No model found in registry: {model_name}")
#         m = max(models, key=lambda x: x.version)
#     else:
#         m = mr.get_model(model_name, version=int(version))      

#     local_dir = m.download()

#     model_path = os.path.join(local_dir, "model.pkl")
#     pipeline = joblib.load(model_path)

#     meta = None
#     meta_path = os.path.join(local_dir, "metadata.json")
#     if os.path.exists(meta_path):
#         with open(meta_path, "r", encoding="utf-8") as f:
#             meta = json.load(f)

#     return pipeline, meta, m
def load_model_from_registry(
    mr,
    model_name: str,
    version: int | None = MODEL_VERSION
):
    """
    mr: project.get_model_registry() 得到的对象，由 app.py 传进来
    model_name: 显式指定模型名（classifier / regressor）
    """
    if version is None:
        models = mr.get_models(model_name)
        if not models:
            raise RuntimeError(f"No model found in registry: {model_name}")
        m = max(models, key=lambda x: x.version)
    else:
        m = mr.get_model(model_name, version=int(version))

    local_dir = m.download()

    model_path = os.path.join(local_dir, "model.pkl")
    pipeline = joblib.load(model_path)

    meta = None
    meta_path = os.path.join(local_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return pipeline, meta, m


# def run_latest_batch_prediction(mr, limit_311: int = 100) -> pd.DataFrame:
#     df_raw = fetch_latest_311(limit=limit_311)
#     if df_raw.empty:
#         return pd.DataFrame()

#     df_311 = preprocess_311_like_training(df_raw)
#     dates = df_311["service_date"].dropna().astype(str).tolist()
#     weather = fetch_weather_daily_for_dates(dates)
#     df_feat = join_weather(df_311, weather)

#     model, meta, model_obj = load_model_from_registry(mr)

#     if meta and "categorical_features" in meta and "numeric_features" in meta:
#         needed_cols = list(meta["categorical_features"]) + list(meta["numeric_features"])
#         for c in needed_cols:
#             if c not in df_feat.columns:
#                 df_feat[c] = pd.NA
#         X_infer = df_feat[needed_cols].copy()
#     else:
#         X_infer = df_feat.copy()

#     proba = model.predict_proba(X_infer)[:, 1]
#     pred = (proba >= 0.5).astype(int)

#     out = df_feat.copy()
#     out["prob_within_48h"] = proba
#     out["pred_label"] = pred
#     out.attrs["model_name"] = getattr(model_obj, "name", model_name if 'model_name' in locals() else MODEL_NAME)
#     out.attrs["model_version"] = getattr(model_obj, "version", None)

def run_latest_batch_prediction(mr, limit_311: int = 100) -> pd.DataFrame:
    # 1. 拉最新 311
    df_raw = fetch_latest_311(limit=limit_311)
    if df_raw.empty:
        return pd.DataFrame()

    # 2. 特征工程（一次）
    df_311 = preprocess_311_like_training(df_raw)
    dates = df_311["service_date"].dropna().astype(str).tolist()
    weather = fetch_weather_daily_for_dates(dates)
    df_feat = join_weather(df_311, weather)

    # 3. 加载两个模型
    clf_model, clf_meta, clf_obj = load_model_from_registry(
        mr, model_name=CLASSIFIER_MODEL_NAME
    )
    reg_model, reg_meta, reg_obj = load_model_from_registry(
        mr, model_name=REGRESSOR_MODEL_NAME
    )

    # 4. 构造模型输入（按各自 metadata）
    def build_X(df, meta):
        if meta and "categorical_features" in meta and "numeric_features" in meta:
            cols = list(meta["categorical_features"]) + list(meta["numeric_features"])
            for c in cols:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[cols].copy()
        return df.copy()

    X_clf = build_X(df_feat, clf_meta)
    X_reg = build_X(df_feat, reg_meta)

    # 5. 预测
    proba = clf_model.predict_proba(X_clf)[:, 1]
    pred_label = (proba >= 0.5).astype(int)

    pred_resolution_hours = reg_model.predict(X_reg)

    # 6. 输出
    out = df_feat.copy()
    out["prob_within_48h"] = proba
    out["pred_within_48h"] = pred_label
    out["pred_resolution_hours"] = pred_resolution_hours

    # 7. 记录模型版本（给 Streamlit 用）
    out.attrs["classifier_model_version"] = clf_obj.version
    out.attrs["regressor_model_version"] = reg_obj.version

    return out


    
    return out
