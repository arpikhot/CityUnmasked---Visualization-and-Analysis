import streamlit as st
from analysis.decay_index import (
    classify_neighborhoods, get_economic_abandonment_zones,
    get_proximity_stats,
    fig_crime_vs_decay_scatter, fig_zone_type_breakdown,
    fig_risk_score_ranking, fig_crime_type_by_zone
)


def render(crime, decay, unfit):
    # st.markdown("### 📉 Urban Decay Index")
    st.caption("Every zip code classified by crime severity and decay level.")
    with st.expander("How are three zone types defined ?"):
        st.markdown("""
        *Type A — Crime-Blight Feedback Zone* 🔴
        High crime AND high decay co-occurring. Both problems must be addressed simultaneously — fixing one without the other breaks only half the cycle.

        *Type B — Economic Abandonment Zone* 🔵
        High decay, LOW crime. Properties are vacant or unfit for economic reasons: landlord failure, population loss, age deterioration. Policing is the WRONG intervention here — investment and ownership reform are needed.

        *Type C — Infrastructure Decay Zone* 🟡
        Unfit-dominant areas driven by structural age and deferred maintenance. Requires code enforcement and rehabilitation funding.

        *Low Risk / Monitoring* ⚫ — below median on both axes.

        *Why this matters:* Type B zones prove blight has multiple causes. Different causes need different solutions.
        """)

    nbr = classify_neighborhoods(crime, decay, unfit)
    abandoned, low_crime_zips = get_economic_abandonment_zones(crime, decay)
    stats = get_proximity_stats(crime)

    # ── Zone KPIs — keep just 4 ──────────────────────────────────
    # nbr = classify_neighborhoods(crime, decay, unfit)
    # abandoned, low_crime_zips = get_economic_abandonment_zones(crime, decay)
    # stats = get_proximity_stats(crime)
    # type_counts = nbr['zone_type'].value_counts()

    # z1, z2, z3, z4 = st.columns(4)
    # for col, (label, val, color) in zip(
    #     [z1, z2, z3, z4],
    #     [
    #         ("Type A — Crime+Blight",   type_counts.get('Type A — Crime-Blight Feedback', 0), "#dc2626"),
    #         ("Type B — Abandonment",    type_counts.get('Type B — Economic Abandonment',  0), "#3b82f6"),
    #         ("Type C — Infrastructure", type_counts.get('Type C — Infrastructure Decay',   0), "#f59e0b"),
    #         ("Low Risk",                type_counts.get('Low Risk / Monitoring',            0), "#6b7280"),
    #     ]
    # ):
    #     with col:
    #         st.markdown(f"""
    #         <div class="metric-card">
    #             <div class="metric-value" style="color:{color}">{val}</div>
    #             <div class="metric-label">{label}</div>
    #         </div>""", unsafe_allow_html=True)

    # st.markdown("---")

    # ── Row 1: Scatter + Zone breakdown ──────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header-red">Crime vs Decay by Zip Code</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_crime_vs_decay_scatter(nbr), use_container_width=True)
        st.caption("Top-right = Type A (crime + blight). High decay + low crime = Type B (economic abandonment).")
    with col2:
        st.markdown('<div class="section-header-red">Top 10 Zip Codes by Risk Score</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_risk_score_ranking(nbr), use_container_width=True)
        st.caption("Risk = crime (40%) + decay (35%) + unresolved violations (25%).")

    # ── Row 2: Proximity stats + Crime by zone ────────────────────
    # st.markdown("---")
    # p1, p2, p3, p4 = st.columns(4)
    # for col, (label, pct, n, color) in zip(
    #     [p1, p2, p3, p4],
    #     [
    #         ("Near Unfit (100m)",  stats['near_unfit_pct'],  stats['near_unfit_n'],  "#f97316"),
    #         ("Near Vacant (100m)", stats['near_vacant_pct'], stats['near_vacant_n'], "#3b82f6"),
    #         ("Near Any Decay",     stats['near_decay_pct'],  stats['near_decay_n'],  "#dc2626"),
    #         ("Near Both",          stats['near_both_pct'],   stats['near_both_n'],   "#7c3aed"),
    #     ]
    # ):
    #     with col:
    #         st.markdown(f"""
    #         <div class="metric-card">
    #             <div class="metric-value" style="color:{color}">{pct}</div>
    #             <div class="metric-label">{label}<br><small>({n:,} crimes)</small></div>
    #         </div>""", unsafe_allow_html=True)

    # st.markdown("---")

    # ── Row 3: Crime types by zone ────────────────────────────────
    st.markdown('<div class="section-header-red">Crime Types by Decay Zone</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig_crime_type_by_zone(crime), use_container_width=True)
    st.caption("Violent crimes cluster in 'Near Both' zones — where unfit and vacant properties overlap.")

    # ── Classification table (collapsed by default) ───────────────
    # with st.expander("📊 Full Zip Code Classification Table"):
    #     display = nbr[['zip_code', 'zone_type', 'crime_count',
    #                     'decay_score', 'pct_unresolved', 'risk_score']].copy()
    #     display['pct_unresolved'] = (display['pct_unresolved'] * 100).round(1).astype(str) + '%'
    #     display['risk_score'] = display['risk_score'].round(1)
    #     display.columns = ['Zip', 'Zone Type', 'Crimes', 'Decay Score', '% Unresolved', 'Risk Score']
    #     st.dataframe(display, use_container_width=True)

    # ── Key finding ───────────────────────────────────────────────
    st.error("🔴 **Key Finding:** ZIP codes 13204, 13205, 13208 lead on both decay and crime. "
             "Type A zones need simultaneous housing AND public safety intervention. "
             f"{len(low_crime_zips)} zip codes show high vacancy but low crime — "
             "economic investment, not policing, is the right solution there.")