"""
Battery Intelligence Platform
================================
Main Streamlit application entry point.
Multi-page navigation with dark theme and premium styling.
"""

import streamlit as st
import os
import sys
import json
from pathlib import Path

# ─── Add current directory to path for imports ─── #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Page Configuration ─── #
st.set_page_config(
    page_title="Battery Intelligence Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Custom CSS ─── #
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─── Sidebar Navigation ─── #
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:20px 0 10px 0">
            <span style="font-size:2.5rem">⚡</span>
            <h1 style="font-size:1.3rem;margin:8px 0 2px 0;
                background:linear-gradient(135deg,#00D1FF,#00FFD5);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                font-weight:800;letter-spacing:-0.5px">
                Battery Intelligence
            </h1>
            <p style="color:#9AA0A6;font-size:0.75rem;margin:0;letter-spacing:0.5px">
                EV BATTERY HEALTH ANALYTICS
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(0,209,255,0.2),transparent);margin:10px 0 20px 0"></div>',
        unsafe_allow_html=True,
    )

    # Navigation uses stable, emoji-free values so routing is never dependent on
    # Unicode rendering or a displayed label.
    page = st.radio(
        "Navigation",
        [
            "Home Dashboard",
            "SoH Prediction",
            "RUL Prediction",
            "Analytics",
            "Explainable AI",
            "Fleet Monitoring",
        ],
        label_visibility="collapsed",
        key="main_navigation",
    )

    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(0,209,255,0.2),transparent);margin:20px 0"></div>',
        unsafe_allow_html=True,
    )

    # Model status
    from app_paths import METADATA_PATH

    if METADATA_PATH.is_file():
        try:
            with METADATA_PATH.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            soh_score = float(meta.get("soh_best_cv_score", 0.0))
            rul_score = float(meta.get("rul_best_cv_score", 0.0))
            st.markdown(
                f"""
                <div style="background:rgba(0,209,255,0.05);border:1px solid rgba(0,209,255,0.15);
                    border-radius:12px;padding:14px;margin-top:8px">
                    <span style="color:#00D1FF;font-weight:600;font-size:0.8rem">MODEL STATUS</span><br>
                    <span style="color:#34D399;font-size:0.75rem">Models available</span><br>
                    <span style="color:#9AA0A6;font-size:0.72rem">
                        SoH: {meta.get('soh_best_model', 'N/A')} (R²={soh_score:.3f})<br>
                        RUL: {meta.get('rul_best_model', 'N/A')} (R²={rul_score:.3f})
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            st.warning("Model metadata could not be read. Run the training pipeline to regenerate artifacts.")
    else:
        st.markdown(
            """
            <div style="background:rgba(248,113,113,0.05);border:1px solid rgba(248,113,113,0.15);
                border-radius:12px;padding:14px;margin-top:8px">
                <span style="color:#F87171;font-weight:600;font-size:0.8rem">MODEL STATUS</span><br>
                <span style="color:#F87171;font-size:0.75rem">Not available</span><br>
                <span style="color:#9AA0A6;font-size:0.72rem">Run the training pipeline first</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """
        <div style="position:fixed;bottom:16px;left:16px;right:16px;text-align:center">
            <span style="color:#5F6368;font-size:0.68rem">
                Battery Intelligence Platform v1.0<br>
                Powered by ML & SHAP
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Page routing
if page == "Home Dashboard":
    import home
    home.render()
elif page == "SoH Prediction":
    import soh_prediction
    soh_prediction.render()
elif page == "RUL Prediction":
    import rul_prediction
    rul_prediction.render()
elif page == "Analytics":
    import analytics
    analytics.render()
elif page == "Explainable AI":
    import explainability_dashboard
    explainability_dashboard.render()
elif page == "Fleet Monitoring":
    import fleet_monitoring
    fleet_monitoring.render()
