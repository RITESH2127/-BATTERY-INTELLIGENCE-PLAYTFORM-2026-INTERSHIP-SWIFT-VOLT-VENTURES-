"""
Battery Intelligence Platform
================================
Main Streamlit application entry point.
Multi-page navigation with dark theme and premium styling.
"""

import streamlit as st
import os

# ─── Page Configuration ─── #
st.set_page_config(
    page_title="Battery Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Custom CSS ─── #
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css")
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

    # Navigation
    page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🔋 SoH Prediction",
            "🔄 RUL Prediction",
            "📊 Analytics",
            "🧠 Explainable AI",
            "🚗 Fleet Monitoring",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(0,209,255,0.2),transparent);margin:20px 0"></div>',
        unsafe_allow_html=True,
    )

    # Model status
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        st.markdown(
            f"""
            <div style="background:rgba(0,209,255,0.05);border:1px solid rgba(0,209,255,0.15);
                border-radius:12px;padding:14px;margin-top:8px">
                <span style="color:#00D1FF;font-weight:600;font-size:0.8rem">🤖 MODEL STATUS</span><br>
                <span style="color:#34D399;font-size:0.75rem">● Models Trained</span><br>
                <span style="color:#9AA0A6;font-size:0.72rem">
                    SoH: {meta.get('soh_best_model', 'N/A')} (R²={meta.get('soh_best_cv_score', 0):.3f})<br>
                    RUL: {meta.get('rul_best_model', 'N/A')} (R²={meta.get('rul_best_cv_score', 0):.3f})
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:rgba(248,113,113,0.05);border:1px solid rgba(248,113,113,0.15);
                border-radius:12px;padding:14px;margin-top:8px">
                <span style="color:#F87171;font-weight:600;font-size:0.8rem">🤖 MODEL STATUS</span><br>
                <span style="color:#F87171;font-size:0.75rem">● Not Trained</span><br>
                <span style="color:#9AA0A6;font-size:0.72rem">Run training pipeline first</span>
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


# ─── Page Routing ─── #
if page == "🏠 Home Dashboard":
    from dashboard import home
    home.render()

elif page == "🔋 SoH Prediction":
    from dashboard import soh_prediction
    soh_prediction.render()

elif page == "🔄 RUL Prediction":
    from dashboard import rul_prediction
    rul_prediction.render()

elif page == "📊 Analytics":
    from dashboard import analytics
    analytics.render()

elif page == "🧠 Explainable AI":
    from dashboard import explainability_dashboard
    explainability_dashboard.render()

elif page == "🚗 Fleet Monitoring":
    from dashboard import fleet_monitoring
    fleet_monitoring.render()
