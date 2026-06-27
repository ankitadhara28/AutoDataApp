# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

from data_engine import (
    get_data_health, get_statistical_summary, suggest_models_logic,
    get_column_summary, get_outliers, get_correlation_matrix,
    suggest_train_test_split, suggest_missing_value_handling
)

st.set_page_config(page_title="Data Profiling Engine", layout="wide")

# ── Minimal Black & White Style ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0A0A0A; color: #F0F0F0; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 3rem 4rem; }

    .card {
        background: #111;
        border: 1px solid #222;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-label { font-size: 0.72rem; color: #555; text-transform: uppercase; letter-spacing: 0.1em; }
    .card-value { font-size: 2.2rem; font-weight: 700; color: #FFF; }

    .chip {
        display: inline-block;
        background: #111;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 5px 12px;
        margin: 4px 4px 4px 0;
        font-size: 0.82rem;
        color: #DDD;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────────────────────
st.title("Automated Data Profiling & Model Recommendation Engine")
st.caption("Upload a CSV to profile your dataset and get model suggestions.")
st.divider()

# ── 1. Upload ─────────────────────────────────────────────────────────────────
st.subheader("1 · Dataset Upload")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.caption(f"{df.shape[0]:,} rows · {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("**Column Summary**")
    st.dataframe(get_column_summary(df), use_container_width=True)
    st.divider()

    # ── 2. Diagnostics ────────────────────────────────────────────────────────
    st.subheader("2 · Data Integrity")
    missing_data, duplicates = get_data_health(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Duplicate Rows</div>
            <div class="card-value">{int(duplicates)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("**Missing values per column**")
        if len(missing_data) > 0:
            st.dataframe(missing_data, use_container_width=True)
            st.markdown("**Suggested Handling Strategy**")
            st.dataframe(suggest_missing_value_handling(df), use_container_width=True)
        else:
            st.success("No missing values found.")

    st.markdown("**Outliers Detected (3σ rule)**")
    outliers = get_outliers(df)
    if len(outliers) > 0:
        st.dataframe(outliers, use_container_width=True)
    else:
        st.success("No outliers detected.")

    st.markdown("**Descriptive Statistics**")
    st.dataframe(get_statistical_summary(df), use_container_width=True)
    st.divider()

    # ── 3. Distribution & Correlation ─────────────────────────────────────────
    st.subheader("3 · Distribution & Correlation Analysis")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        col = st.selectbox("Select a numeric column:", numeric_cols)
        fig = px.histogram(df, x=col, color_discrete_sequence=["#FFFFFF"], template="simple_white")
        fig.update_layout(
            plot_bgcolor="#0F0F0F", paper_bgcolor="#0F0F0F",
            font=dict(color="#888"), margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Correlation Heatmap**")
        corr = get_correlation_matrix(df)
        heatmap = px.imshow(corr, color_continuous_scale="Greys", text_auto=True)
        heatmap.update_layout(
            plot_bgcolor="#0F0F0F", paper_bgcolor="#0F0F0F",
            font=dict(color="#888"), margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("No numeric columns to visualise.")
    st.divider()

    # ── 4. Model Recommendation ───────────────────────────────────────────────
    st.subheader("4 · Model Recommendation")
    target = st.selectbox("Select target variable (Y):", df.columns.tolist())

    if target:
        prob_type, models = suggest_models_logic(df, target)
        st.markdown(f"**Problem type:** `{prob_type}`")

        split, reason = suggest_train_test_split(df)
        st.markdown(f"**Suggested train/test split:** `{split}` — {reason}")

        st.markdown("**Suggested models:**")
        chips = "".join(f'<span class="chip">{m}</span>' for m in models)
        st.markdown(chips, unsafe_allow_html=True)