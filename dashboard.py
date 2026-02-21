import streamlit as st
from streamlit_folium import st_folium
from crime_unfit_analysis import (
    load_data, get_kpis,
    fig_top_crimes, fig_qol_pie, fig_crime_by_month,
    fig_crime_by_hour, fig_near_vs_not,
    fig_unfit_by_year, fig_open_closed_pie,
    fig_unfit_by_zip, fig_open_by_zip,
    build_map, fig_prediction
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CityUnmasked — Syracuse Urban Analysis",
    page_icon="🏙️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e; color: white; padding: 20px;
        border-radius: 10px; text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #f97316; }
    .metric-label { font-size: 0.9rem; color: #aaa; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem; font-weight: bold;
        border-left: 4px solid #f97316;
        padding-left: 10px; margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
crime, crime_2024, unfit, unfit_clean = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏙️ CityUnmasked — Syracuse Crime & Urban Decay Analysis")
st.markdown("**Track 3 — Urban Data Analysis | City of Syracuse Datathon 2026**")
st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
kpis = get_kpis(crime_2024, unfit)
k1, k2, k3, k4, k5 = st.columns(5)
for col, (label, val) in zip(
    [k1, k2, k3, k4, k5],
    [("2024 Crime Incidents",       kpis["total_crimes"]),
     ("Distinct Crime Types",       kpis["crime_types"]),
     ("Unfit Properties All Time",  kpis["total_unfit"]),
     ("Still Open Violations",      kpis["open_violations"]),
     ("Crimes Near Unfit (100m)",   kpis["pct_near_unfit"])]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Crime Analysis",
    "🏚️ Unfit Properties",
    "🗺️ Map",
    "🔮 Prediction"
])

# ── Tab 1 — Crime Analysis ────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Top Crime Types in 2024</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_top_crimes(crime_2024), use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Serious vs Quality-of-Life Crimes</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_qol_pie(crime_2024), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Crime by Month (2024)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_crime_by_month(crime_2024), use_container_width=True)
    with col4:
        st.markdown('<div class="section-header">Crime by Hour of Day</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_crime_by_hour(crime_2024), use_container_width=True)

    st.markdown('<div class="section-header">Crime Types: Near vs Not Near Unfit Properties (100m)</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_near_vs_not(crime_2024), use_container_width=True)
    st.info("💡 **Insight:** Areas within 100m of unfit properties represent ~12% of city area but account for 27% of all crimes — roughly **2x the expected rate.**")

# ── Tab 2 — Unfit Properties ──────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Violations Filed Per Year</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_unfit_by_year(unfit), use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Open vs Closed Violations</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_open_closed_pie(unfit), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Top Zip Codes by Unfit Properties</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_unfit_by_zip(unfit), use_container_width=True)
    with col4:
        st.markdown('<div class="section-header">Open Violations by Zip Code</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_open_by_zip(unfit), use_container_width=True)

    st.warning("⚠️ **Trend Alert:** Unfit violations grew **33x from 2014 to 2025**, with 73% still unresolved. Steepest acceleration began in 2022.")

# ── Tab 3 — Map ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Crime Heatmap + Unfit Property Locations</div>', unsafe_allow_html=True)
    st.caption("🔴 Open unfit property  |  ⚫ Closed unfit property  |  🌡️ Heatmap = crime density (2024)")
    st_folium(build_map(crime_2024, unfit_clean), width=1100, height=550)
    st.success("🗺️ Open unfit properties (red) visually cluster inside the highest-crime zones — particularly the northwest and southwest corridors of Syracuse.")

# ── Tab 4 — Prediction ────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Predicting Unfit Property Violations — Next 3 Years</div>', unsafe_allow_html=True)
    fig, years, preds = fig_prediction(unfit)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, yr, pred in zip([c1, c2, c3], years, preds):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pred}</div>
                <div class="metric-label">Predicted violations in {yr}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("###")
    st.error("📈 If current trends continue, Syracuse could see **100+ new violations per year** through 2027, compounding the backlog of 187 already-open cases.")
    st.markdown("---")
    st.markdown("### 🎯 Policy Recommendations")
    st.markdown("""
    1. **Prioritize zip codes 13204, 13205, 13208** — majority of both unfit properties and crime hotspots
    2. **Fast-track open violations** — 73% remain unresolved; faster resolution may reduce nearby crime
    3. **Increase enforcement capacity** — violations are growing 10x faster than closures
    4. **Monitor 100m buffer zones** — proactive policing near open unfit properties could reduce the 2x crime concentration
    """)