"""
Slope Stability FOS Predictor - Multi-Model
--------------------------------------------
A Streamlit GUI that lets you choose from 14 trained regression models
to predict the Factor of Safety (FOS) of a slope, and shows a
performance comparison table (R2_mean) for all models.

HOW TO RUN
1. Place this script (slope_fos_app.py) in the SAME folder as all
   14 .joblib model files, i.e.:
   C:\\Users\\satya\\Documents\\python_codes\\All codes\\Slope Stability

2. Activate your environment (pinn_env) and make sure these are installed:
       pip install streamlit joblib pandas numpy scikit-learn xgboost lightgbm catboost

3. Run:
       streamlit run slope_fos_app.py

MODEL INPUTS (order the models were trained on):
    gama  -> Unit weight of soil (kN/m3)
    c     -> Cohesion
    phi   -> Friction angle (degrees)
    beta  -> Slope angle (degrees)
    H     -> Slope height (m)
    ru    -> Pore pressure ratio (0-1)
"""

import os
import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Slope Stability FOS Predictor",
    page_icon="\u26f0\ufe0f",
    layout="centered",
)

FEATURE_ORDER = ["gama", "c", "phi", "beta", "H", "ru"]

# Model display name -> (filename, R2_mean from your validation results)
MODEL_INFO = {
    "DT":        {"file": "trained_dt_model.joblib",       "r2": 0.806},
    "KNN":       {"file": "trained_knn_model.joblib",      "r2": 0.849},
    "SVM":       {"file": "trained_svm_model.joblib",      "r2": 0.808},
    "Voting":    {"file": "trained_voting_model.joblib",   "r2": 0.805},
    "Stacking":  {"file": "trained_stacking_model.joblib", "r2": 0.843},
    "Bagging":   {"file": "trained_bagging_model.joblib",  "r2": 0.867},
    "RF":        {"file": "trained_RF_model.joblib",       "r2": 0.869},
    "ET":        {"file": "trained_ET_model.joblib",       "r2": 0.886},
    "XGBoost":   {"file": "trained_XGB_model.joblib",      "r2": 0.866},
    "AdaBoost":  {"file": "trained_AdaBoost_model.joblib", "r2": 0.668},
    "GB":        {"file": "trained_GB_model.joblib",       "r2": 0.848},
    "HGB":       {"file": "trained_HGB_model.joblib",      "r2": 0.868},
    "LightGBM":  {"file": "trained_LightGBM_model.joblib", "r2": 0.876},
    "CatBoost":  {"file": "trained_CatBoost_model.joblib", "r2": 0.867},
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("\u26f0\ufe0f Slope Stability FOS Predictor")
st.caption("Choose a model, enter slope parameters, and predict the Factor of Safety")

# ----------------------------------------------------------------------
# Performance comparison table
# ----------------------------------------------------------------------
st.subheader("Model Performance Comparison")

perf_df = pd.DataFrame(
    [{"Model": name, "R2_mean": info["r2"]} for name, info in MODEL_INFO.items()]
).sort_values("R2_mean", ascending=False).reset_index(drop=True)

st.dataframe(
    perf_df.style.format({"R2_mean": "{:.3f}"}).background_gradient(
        subset=["R2_mean"], cmap="Greens"
    ),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ----------------------------------------------------------------------
# Model loading (cached per model file so switching is fast after first use)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


st.subheader("Choose a Model")

model_names_sorted_by_r2 = perf_df["Model"].tolist()
selected_name = st.selectbox(
    "Model",
    options=model_names_sorted_by_r2,
    index=model_names_sorted_by_r2.index("ET") if "ET" in model_names_sorted_by_r2 else 0,
    help="Models are listed best-to-worst by R2_mean.",
)

selected_info = MODEL_INFO[selected_name]
selected_path = os.path.join(SCRIPT_DIR, selected_info["file"])

model = None
model_load_error = None

if os.path.exists(selected_path):
    try:
        model = load_model(selected_path)
    except Exception as e:
        model_load_error = str(e)
else:
    model_load_error = f"File not found next to this script: {selected_info['file']}"

if model is not None:
    st.success(f"**{selected_name}** loaded (R2_mean = {selected_info['r2']:.3f})")
else:
    st.error(
        f"Couldn't load **{selected_name}**: {model_load_error}\n\n"
        "This is usually a library version mismatch (e.g. scikit-learn/xgboost/"
        "lightgbm/catboost version differs from the one used to train the model). "
        "Make sure pinn_env has the same package versions used during training."
    )

st.divider()

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
st.subheader("Slope Parameters")

col1, col2 = st.columns(2)

with col1:
    gama = st.number_input(
        "Unit weight, \u03b3 (kN/m\u00b3)",
        min_value=0.0, max_value=30.0, value=18.0, step=0.1,
        help="Bulk/unit weight of the soil forming the slope.",
    )
    phi = st.number_input(
        "Friction angle, \u03c6 (degrees)",
        min_value=0.0, max_value=50.0, value=25.0, step=0.5,
        help="Angle of internal friction of the soil.",
    )
    H = st.number_input(
        "Slope height, H (m)",
        min_value=0.1, max_value=200.0, value=10.0, step=0.5,
        help="Vertical height of the slope.",
    )

with col2:
    c = st.number_input(
        "Cohesion, c",
        min_value=0.0, max_value=200.0, value=10.0, step=0.5,
        help="Cohesion of the soil. Use the same unit as during training (e.g. kPa or kg/cm\u00b2).",
    )
    beta = st.number_input(
        "Slope angle, \u03b2 (degrees)",
        min_value=1.0, max_value=90.0, value=30.0, step=0.5,
        help="Inclination of the slope face from horizontal.",
    )
    ru = st.number_input(
        "Pore pressure ratio, ru",
        min_value=0.0, max_value=1.0, value=0.2, step=0.01,
        help="Dimensionless pore water pressure ratio (0 = dry, higher = more saturated).",
    )

st.divider()

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
predict_clicked = st.button(
    "Predict FOS", type="primary", use_container_width=True, disabled=(model is None)
)

if predict_clicked and model is not None:
    input_df = pd.DataFrame([{
        "gama": gama, "c": c, "phi": phi, "beta": beta, "H": H, "ru": ru,
    }])[FEATURE_ORDER]

    try:
        fos_pred = float(model.predict(input_df)[0])

        st.subheader("Result")
        st.metric(f"Predicted FOS ({selected_name})", f"{fos_pred:.3f}")

        # Indicative interpretation only - adjust thresholds to your project's
        # governing standard/criteria if different.
        if fos_pred < 1.0:
            st.error("FOS < 1.0 \u2192 Slope is predicted to be UNSTABLE.")
        elif fos_pred < 1.3:
            st.warning("1.0 \u2264 FOS < 1.3 \u2192 Marginally stable. Review design.")
        else:
            st.success("FOS \u2265 1.3 \u2192 Slope is predicted to be STABLE.")

        with st.expander("Show input values sent to the model"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed with {selected_name}: {e}")

st.divider()
st.caption(
    "Note: The stability thresholds shown above (1.0 / 1.3) are common "
    "geotechnical rules of thumb, not part of any trained model. R2_mean "
    "values in the table above are the validation results you supplied "
    "and are not recomputed by this app."
)