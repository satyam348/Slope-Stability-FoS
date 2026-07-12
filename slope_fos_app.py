"""
Slope Stability FOS Predictor - Multi-Model
--------------------------------------------
A Streamlit GUI that lets you choose from 14 trained regression models
to predict the Factor of Safety (FOS) of a slope. Includes a performance
comparison table/chart, a schematic slope diagram based on your inputs,
and a visual FOS gauge.

HOW TO RUN
1. Place this script (slope_fos_app.py) in the SAME folder as all
   14 model .joblib files AND their matching *_scaler.joblib files, i.e.:
   C:\\Users\\satya\\Documents\\python_codes\\All codes\\Slope Stability

2. Activate your environment (pinn_env) and make sure these are installed:
       pip install streamlit joblib pandas numpy matplotlib scikit-learn xgboost lightgbm catboost

3. Run:
       streamlit run slope_fos_app.py

MODEL INPUTS (order the models were trained on):
    gama  -> Unit weight of soil (kN/m3)
    c     -> Cohesion
    phi   -> Friction angle (degrees)
    beta  -> Slope angle (degrees)
    H     -> Slope height (m)
    ru    -> Pore pressure ratio (0-1)

IMPORTANT: All 14 models were trained on SCALED features (StandardScaler
fit on the full dataset, then the model fit on the scaled result). This
app therefore loads and applies a matching *_scaler.joblib for every
model - including tree-based ones - before predicting.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Slope Stability FOS Predictor",
    page_icon="\u26f0\ufe0f",
    layout="centered",
)

FEATURE_ORDER = ["gama", "c", "phi", "beta", "H", "ru"]

# Model display name -> model file, scaler file, R2_mean from your validation results
MODEL_INFO = {
    "DT":        {"file": "trained_dt_model.joblib",       "scaler": "trained_dt_scaler.joblib",       "r2": 0.806},
    "KNN":       {"file": "trained_knn_model.joblib",      "scaler": "trained_knn_scaler.joblib",      "r2": 0.849},
    "SVM":       {"file": "trained_svm_model.joblib",      "scaler": "trained_svm_scaler.joblib",      "r2": 0.808},
    "Voting":    {"file": "trained_voting_model.joblib",   "scaler": "trained_voting_scaler.joblib",   "r2": 0.805},
    "Stacking":  {"file": "trained_stacking_model.joblib", "scaler": "trained_stacking_scaler.joblib", "r2": 0.843},
    "Bagging":   {"file": "trained_Bagging_model.joblib",  "scaler": "trained_Bagging_scaler.joblib",  "r2": 0.867},
    "RF":        {"file": "trained_RF_model.joblib",       "scaler": "trained_RF_scaler.joblib",       "r2": 0.869},
    "ET":        {"file": "trained_ET_model.joblib",       "scaler": "trained_ET_scaler.joblib",       "r2": 0.886},
    "XGBoost":   {"file": "trained_XGBoost_model.joblib",  "scaler": "trained_XGBoost_scaler.joblib",  "r2": 0.866},
    "AdaBoost":  {"file": "trained_AdaBoost_model.joblib", "scaler": "trained_AdaBoost_scaler.joblib", "r2": 0.668},
    "GB":        {"file": "trained_GB_model.joblib",       "scaler": "trained_GB_scaler.joblib",       "r2": 0.848},
    "HGB":       {"file": "trained_HGB_model.joblib",      "scaler": "trained_HGB_scaler.joblib",      "r2": 0.868},
    "LightGBM":  {"file": "trained_LightGBM_model.joblib", "scaler": "trained_LightGBM_scaler.joblib", "r2": 0.876},
    "CatBoost":  {"file": "trained_CatBoost_model.joblib", "scaler": "trained_CatBoost_scaler.joblib", "r2": 0.867},
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("\u26f0\ufe0f Slope Stability FoS Predictor GUI")
st.caption("Choose a model, enter slope parameters, and predict the Factor of Safety")




st.divider()

# ----------------------------------------------------------------------
# Model + scaler loading (cached so switching models is fast after first use)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_joblib(path: str):
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
model_path = os.path.join(SCRIPT_DIR, selected_info["file"])
scaler_path = os.path.join(SCRIPT_DIR, selected_info["scaler"])

model = None
scaler = None
load_error = None

if not os.path.exists(model_path):
    load_error = f"Model file not found next to this script: {selected_info['file']}"
elif not os.path.exists(scaler_path):
    load_error = (
        f"Scaler file not found next to this script: {selected_info['scaler']}. "
        "This model was trained on scaled features, so predictions without the "
        "matching scaler will be incorrect."
    )
else:
    try:
        model = load_joblib(model_path)
        scaler = load_joblib(scaler_path)
    except Exception as e:
        load_error = str(e)

if model is not None and scaler is not None:
    st.success(f"**{selected_name}** loaded (R2_mean = {selected_info['r2']:.3f})")
else:
    st.error(
        f"Couldn't load **{selected_name}**: {load_error}\n\n"
        "If this is a library-version error, make sure pinn_env has the same "
        "scikit-learn/xgboost/lightgbm/catboost versions used during training."
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

# ----------------------------------------------------------------------
# Slope schematic diagram (updates live with inputs)
# ----------------------------------------------------------------------
def draw_slope_diagram(H, beta_deg, ru, gama, c, phi):
    """Draws a simple 2D cross-section schematic of the slope geometry."""
    beta_rad = np.deg2rad(beta_deg)
    run = H / np.tan(beta_rad) if np.tan(beta_rad) > 1e-6 else H * 5

    # Ground/slope profile: flat toe -> slope face -> flat crest
    toe_extension = max(run * 0.6, 3)
    crest_extension = max(run * 0.8, 4)

    x = [-toe_extension, 0, run, run + crest_extension]
    y = [0, 0, H, H]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Slope body (fill)
    poly_x = x + [run + crest_extension, -toe_extension]
    poly_y = y + [-2, -2]
    ax.fill(poly_x, poly_y, color="#c8a165", alpha=0.85, edgecolor="black", linewidth=1.2, zorder=2)

    # Ground/slope line
    ax.plot(x, y, color="black", linewidth=2, zorder=3)

    # Phreatic (water) line based on ru - approximate depiction, height scaled by ru
    water_h = H * min(ru * 1.6, 0.95)
    wx = [-toe_extension, run * 0.15, run * 0.85, run + crest_extension]
    wy = [max(water_h - H, -2) * 0 + water_h * 0.15,
          water_h * 0.6, water_h, water_h]
    wy = [min(v, H - 0.3) for v in wy]
    ax.plot(wx, wy, color="#1f77b4", linewidth=1.8, linestyle="--", zorder=4, label="Approx. phreatic line")
    ax.fill_between(wx, [-2, -2, -2, -2], wy, color="#7fb8e0", alpha=0.35, zorder=1)

    # Height dimension line
    ax.annotate("", xy=(run + crest_extension * 0.35, 0), xytext=(run + crest_extension * 0.35, H),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1))
    ax.text(run + crest_extension * 0.35 + 0.5, H / 2, f"H = {H:.1f} m",
            rotation=90, va="center", fontsize=10)

    # Slope angle annotation
    ax.text(run * 0.35, H * 0.35, f"\u03b2 = {beta_deg:.1f}\u00b0", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.85))

    # Soil property label box
    props_text = (
        f"$\\gamma$ = {gama:.1f} kN/m$^3$\n"
        f"c = {c:.1f}\n"
        f"$\\phi$ = {phi:.1f}\u00b0\n"
        f"$r_u$ = {ru:.2f}"
    )
    ax.text(-toe_extension * 0.95, H * 0.7, props_text, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e6", edgecolor="gray"))

    ax.set_xlim(-toe_extension * 1.1, run + crest_extension * 1.1)
    ax.set_ylim(-2, H * 1.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Horizontal distance (m)", fontsize=10)
    ax.set_ylabel("Elevation (m)", fontsize=10)
    ax.set_title("Slope Cross-Section (schematic)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    plt.tight_layout()
    return fig


st.subheader("Slope Geometry Preview")
slope_fig = draw_slope_diagram(H, beta, ru, gama, c, phi)
st.pyplot(slope_fig)
plt.close(slope_fig)
st.caption(
    "Schematic only - proportions are illustrative, not an engineering-scale drawing. "
    "The dashed line is an approximate phreatic surface implied by ru, not a computed seepage solution."
)

st.divider()

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
predict_clicked = st.button(
    "Predict FOS", type="primary", use_container_width=True,
    disabled=(model is None or scaler is None),
)

def draw_fos_gauge(fos_value):
    fig, ax = plt.subplots(figsize=(8, 1.6))
    max_scale = max(2.0, fos_value * 1.2)

    ax.axvspan(0, 1.0, color="#e74c3c", alpha=0.35)
    ax.axvspan(1.0, 1.3, color="#f39c12", alpha=0.35)
    ax.axvspan(1.3, max_scale, color="#2ecc71", alpha=0.35)

    ax.axvline(fos_value, color="black", linewidth=2.5, zorder=5)
    ax.plot(fos_value, 0.5, marker="v", color="black", markersize=12, zorder=6)
    ax.text(fos_value, 1.15, f"FOS = {fos_value:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, max_scale)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xlabel("Factor of Safety", fontsize=10)
    ax.set_title("Unstable | Marginal | Stable", fontsize=10)
    plt.tight_layout()
    return fig


if predict_clicked and model is not None and scaler is not None:
    input_df = pd.DataFrame([{
        "gama": gama, "c": c, "phi": phi, "beta": beta, "H": H, "ru": ru,
    }])[FEATURE_ORDER]

    try:
        input_scaled = scaler.transform(input_df)
        fos_pred = float(model.predict(input_scaled)[0])

        st.subheader("Result")
        st.metric(f"Predicted FOS ({selected_name})", f"{fos_pred:.3f}")

        gauge_fig = draw_fos_gauge(fos_pred)
        st.pyplot(gauge_fig)
        plt.close(gauge_fig)

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
            st.caption("Values shown are raw inputs; the model was fed the scaled version internally.")

    except Exception as e:
        st.error(f"Prediction failed with {selected_name}: {e}")


tab_table, tab_chart = st.tabs(["Table", "Chart"])

with tab_table:
    st.dataframe(
        perf_df.style.format({"R2_mean": "{:.3f}"}).background_gradient(
            subset=["R2_mean"], cmap="Greens"
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
# ----------------------------------------------------------------------
# Performance comparison table + chart
# ----------------------------------------------------------------------
st.subheader("Model Performance Comparison")

perf_df = pd.DataFrame(
    [{"Model": name, "R2_mean": info["r2"]} for name, info in MODEL_INFO.items()]
).sort_values("R2_mean", ascending=False).reset_index(drop=True)
with tab_chart:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Greens(np.linspace(0.4, 0.95, len(perf_df)))
    bars = ax.barh(perf_df["Model"], perf_df["R2_mean"], color=colors, edgecolor="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("R\u00b2 mean", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_title("Model Performance (R\u00b2 mean)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, perf_df["R2_mean"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                 va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption(
    "Note: The stability thresholds shown above (1.0 / 1.3) are common "
    "geotechnical rules of thumb, not part of any trained model. R2_mean "
    "values in the table above are the validation results you supplied "
    "and are not recomputed by this app. The slope diagram is a schematic "
    "visualization, not a scaled engineering drawing or slip-surface analysis."
)