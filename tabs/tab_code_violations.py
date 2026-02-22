import streamlit as st
from analysis.models import (
    run_granger_causality, fig_granger_pvalues, fig_granger_timeseries
)
from analysis.code_violations import (
    fig_violations_by_year_tier, fig_tier_pie,
    fig_violations_by_zip, fig_violations_by_neighborhood
)


def render(crime, cv):
    # ── Title & high-level framing ──────────────────────────────────────────────
    # st.markdown("### 🧱 Code Violations & Physical Decay")
    st.caption("Where physical neglect is concentrated, how severe it is, and whether it predicts crime spikes.")
    # st.caption(
    #     f"{len(cv):,} physical decay violations "
    #     f"({cv['year'].min()}–{cv['year'].max()}) — admin-only issues removed; "
    #     "this tab is **physical decay only**."
    # )

    # ── Tier explanation (collapsed) ────────────────────────────────────────────
    with st.expander("ℹ️ What are Tier 1 / 2 / 3 violations?"):
        st.markdown(
            """
**Tier 1 — Structural / Critical** 🔴  
Foundation, roof, collapse risk, unfit for habitation.

**Tier 2 — Systems Failure** 🟠  
Plumbing, electrical, heating, life-safety systems.

**Tier 3 — Environmental Neglect** 🟡  
Overgrowth, trash, exterior blight and visible abandonment.
"""
        )

    # ── KPI row ────────────────────────────────────────────────────────────────
    # k1, k2, k3, k4 = st.columns(4)

    # structural = (cv["tier"] == 3).sum()
    # structural_pct = structural / len(cv) * 100

    # kpi_data = [
    #     ("Total physical violations", f"{len(cv):,}", "#dc2626"),
    #     ("Still open", f"{cv['is_open'].sum():,} ({cv['is_open'].mean()*100:.0f}%)", "#f97316"),
    #     ("Structural / critical", f"{structural:,} ({structural_pct:.0f}%)", "#dc2626"),
    #     ("Years of data", f"{cv['year'].min()}–{cv['year'].max()}", "#6b7280"),
    # ]

    # for col, (label, val, color) in zip([k1, k2, k3, k4], kpi_data):
    #     with col:
    #         st.markdown(
    #             f"""
    #         <div class="metric-card">
    #             <div class="metric-value" style="color:{color}">{val}</div>
    #             <div class="metric-label">{label}</div>
    #         </div>""",
    #             unsafe_allow_html=True,
    #         )

    # st.markdown("---")

    # ── Row 1: Trend + composition ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="section-header-red">Top ZIP Codes by Violations</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_violations_by_zip(cv), use_container_width=True)
        st.caption(
            "📌 13205, 13204, 13208 dominate — the same hotspots that appear in crime, unfit housing, and vacancy."
        )


    with col2:
        st.markdown(
            '<div class="section-header-red">Violation Mix by Tier</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_tier_pie(cv), use_container_width=True)
        st.caption("Share of all violations that are structural, systems-related, or environmental neglect.")

    # ── Row 2: Where the problem is ───────────────────────────────────────────
    # st.markdown("---")
    # col3, col4 = st.columns(2)
    # with col3:
    #     st.markdown(
    #         '<div class="section-header-red">Top ZIP Codes by Violations</div>',
    #         unsafe_allow_html=True,
    #     )
    #     st.plotly_chart(fig_violations_by_zip(cv), use_container_width=True)
    #     st.caption(
    #         "📌 13205, 13204, 13208 dominate — the same hotspots that appear in crime, unfit housing, and vacancy."
    #     )

    # with col4:
    #     with st.expander("🏘️ Top Neighborhoods by Violations", expanded=False):
    #         st.markdown(
    #             '<div class="section-header-red">Top Neighborhoods</div>',
    #             unsafe_allow_html=True,
    #         )
    #         st.plotly_chart(fig_violations_by_neighborhood(cv), use_container_width=True)
    #         st.caption(
    #             "📌 Northside, Brighton, Near Westside — same neighborhoods as the crime heatmap and vacancy analysis."
    #         )

    # ── Granger causality ─────────────────────────────────────────────────────
    st.divider()
    # st.markdown("### 📈 Granger Causality — Do Violations Predict Crime? (108 Months)")
    # st.caption(
    #     "Bidirectional test using 92,790 code violations across 108 months. "
    #     "This is the statistically powerful version — far stronger than a 24-month series."
    # )

#     with st.expander("ℹ️ What does bidirectional Granger testing tell us?"):
#         st.markdown(
#             """
# **Violations → Crime**  
# Past violations predict future crime → physical decay is an early-warning signal.

# **Crime → Violations**  
# Past crime predicts future violations → crime drives abandonment and neglect.

# **Both significant**  
# Feedback loop: each accelerates the other — justification for simultaneous Type A intervention.

# **Neither significant**  
# Relationship may be contemporaneous or driven by a shared cause like poverty or disinvestment.
# """
#         )

#     # Adjust this unpacking if your actual function returns a different tuple
    granger_results, _, ts, interpretation = run_granger_causality(crime, cv)

    # if interpretation:
    #     st.info(f"**Result:** {interpretation}")

    if granger_results is not None:
        # Hero chart: p-values
        fig_gc = fig_granger_pvalues(granger_results)
        if fig_gc:
            st.plotly_chart(fig_gc, use_container_width=True)
            st.caption(
                "📌 Bars below the red line are statistically significant. "
                "Orange = violations predict crime; blue = crime predicts violations."
            )

        # Extra diagnostics in an expander
    #     with st.expander("📊 Full Granger Diagnostics (Time Series + Table)"):
    #         fig_ts = fig_granger_timeseries(ts)
    #         if fig_ts:
    #             st.plotly_chart(fig_ts, use_container_width=True)
    #             st.caption(
    #                 "📌 Monthly crime vs code violations over nine years. "
    #                 "Violation spikes preceding crime spikes support the causal direction."
    #             )
    #         st.dataframe(granger_results, use_container_width=True)

    # # ── Closing insight ───────────────────────────────────────────────────────
    # st.info(
    #     "Code violations act as an early-warning system: ZIP codes with chronic structural neglect "
    #     "also appear as high-risk zones in the Crime and Decay Index tabs."
    # )