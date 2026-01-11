# app.py
import streamlit as st
import hopsworks

from inference_pipeline import run_latest_batch_prediction


HOPSWORKS_API_KEY = ""
HOPSWORKS_HOST = "c.app.hopsworks.ai"
HOPSWORKS_PROJECT = "Lab1_xiaotong"
# st.set_page_config(page_title="NYC 311 Prediction", layout="wide")
# st.title("NYC 311 Resolution Prediction (Latest Batch)")

# limit_311 = st.number_input("Max latest 311 records", min_value=1, max_value=100, value=100)

# def get_mr():
#     project = hopsworks.login(
#     api_key_value="",  
#     host="c.app.hopsworks.ai",  
#     project="Lab1_xiaotong", 
#     engine="python"  
#     )
#     return project.get_model_registry()

# if st.button("Run Latest Batch Prediction"):
#     with st.spinner("Connecting to Hopsworks..."):
#         mr = get_mr()

#     with st.spinner("Fetching latest 311 + weather, preprocessing, downloading model, and predicting..."):
#         df_out = run_latest_batch_prediction(mr, limit_311=int(limit_311))

#     if df_out.empty:
#         st.warning("No data returned from 311 API.")
#     else:
#         ver = df_out.attrs.get("model_version", None)
#         st.success(f"Done. Model version: {ver if ver is not None else 'unknown'}")

#         st.subheader("Summary")
#         st.write({
#             "rows": int(len(df_out)),
#             "avg_prob": float(df_out["prob_within_48h"].mean()),
#             "pred_label_1_ratio": float((df_out["pred_label"] == 1).mean()),
#         })

#         st.subheader("Predictions")
#         st.dataframe(df_out, use_container_width=True)

#         csv = df_out.to_csv(index=False).encode("utf-8-sig")
#         st.download_button(
#             "Download predictions as CSV",
#             data=csv,
#             file_name="latest_311_predictions.csv",
#             mime="text/csv",

#         )


# app.py
import streamlit as st
import hopsworks
from inference_pipeline import run_latest_batch_prediction

st.set_page_config(page_title="NYC 311 Prediction", layout="wide")
st.title("NYC 311 Resolution Prediction (Latest Batch)")

limit_311 = st.number_input(
    "Max latest 311 records",
    min_value=1,
    max_value=100,
    value=100
)

def get_mr():
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        engine="python",
    )
    return project.get_model_registry()

if st.button("Run Latest Batch Prediction"):
    with st.spinner("Connecting to Hopsworks..."):
        mr = get_mr()

    with st.spinner("Running batch inference (classification + regression)..."):
        df_out = run_latest_batch_prediction(mr, limit_311=int(limit_311))

    if df_out.empty:
        st.warning("No data returned from 311 API.")
        st.stop()

    clf_ver = df_out.attrs.get("classifier_model_version")
    reg_ver = df_out.attrs.get("regressor_model_version")

    st.success(
        f"Done. Classifier v{clf_ver if clf_ver else 'unknown'} | "
        f"Regressor v{reg_ver if reg_ver else 'unknown'}"
    )

    st.subheader("Summary")
    st.write({
        "rows": int(len(df_out)),
        "avg_prob_within_48h": float(df_out["prob_within_48h"].mean()),
        "pred_within_48h_ratio": float(df_out["pred_within_48h"].mean()),
        "avg_pred_resolution_hours": float(df_out["pred_resolution_hours"].mean()),
    })

    st.subheader("Predictions")
    st.dataframe(
        df_out[
            [
                "created_date",
                "agency",
                "complaint_type",
                "borough",
                "prob_within_48h",
                "pred_within_48h",
                "pred_resolution_hours",
            ]
        ],
        use_container_width=True,
    )

    csv = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download predictions as CSV",
        data=csv,
        file_name="latest_311_predictions.csv",
        mime="text/csv",
    )
