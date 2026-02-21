import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from sklearn.neighbors import BallTree
from sklearn.linear_model import LinearRegression

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    crime = pd.read_csv("crime_2024.csv")
    unfit = pd.read_csv("Unfit_Properties.csv")

    crime['DATEEND']     = pd.to_datetime(crime['DATEEND'], format='mixed', utc=True)
    crime['year_n']      = crime['DATEEND'].dt.year
    crime['month']       = crime['DATEEND'].dt.month
    crime['month_name']  = crime['DATEEND'].dt.strftime('%b')
    crime['hour']        = crime['TIMESTART'].astype(str).str.zfill(4).str[:2].astype(int)
    crime                = crime.dropna(subset=['LAT', 'LONG'])

    unfit['violation_date'] = pd.to_datetime(unfit['violation_date'], format='mixed', utc=True)
    unfit['year']           = unfit['violation_date'].dt.year
    unfit_clean             = unfit.dropna(subset=['Latitude', 'Longitude'])

    crime_2024 = crime[crime['year_n'] == 2024].copy()

    # Spatial join: crimes within 100m of an unfit property
    c_coords = np.radians(crime_2024[['LAT', 'LONG']].values)
    u_coords = np.radians(unfit_clean[['Latitude', 'Longitude']].values)
    tree     = BallTree(u_coords, metric='haversine')
    counts   = tree.query_radius(c_coords, r=100/6_371_000, count_only=True)
    crime_2024['near_unfit'] = counts > 0

    return crime, crime_2024, unfit, unfit_clean

# ══════════════════════════════════════════════════════════════════════════════
# KPI METRICS
# ══════════════════════════════════════════════════════════════════════════════
def get_kpis(crime_2024, unfit):
    return {
        "total_crimes"    : len(crime_2024),
        "crime_types"     : crime_2024['CODE_DEFINED'].nunique(),
        "total_unfit"     : len(unfit),
        "open_violations" : int((unfit['status_type_name'] == 'Open').sum()),
        "pct_near_unfit"  : f"{crime_2024['near_unfit'].mean()*100:.0f}%"
    }

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CRIME ANALYSIS CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def fig_top_crimes(crime_2024):
    top = crime_2024['CODE_DEFINED'].value_counts().head(8).reset_index()
    top.columns = ['Crime Type', 'Count']
    fig = px.bar(top, x='Count', y='Crime Type', orientation='h',
                 color='Count', color_continuous_scale='Reds')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      coloraxis_showscale=False, height=380)
    return fig

def fig_qol_pie(crime_2024):
    qol = crime_2024['QualityOfLife'].map(
        {True: 'Quality of Life', False: 'Serious Crime'}).value_counts()
    fig = px.pie(values=qol.values, names=qol.index,
                 color_discrete_sequence=['#f97316', '#334155'], hole=0.45)
    fig.update_layout(height=380)
    return fig

def fig_crime_by_month(crime_2024):
    month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    monthly = crime_2024.groupby('month_name').size()\
                        .reindex(month_order, fill_value=0).reset_index()
    monthly.columns = ['Month', 'Count']
    fig = px.line(monthly, x='Month', y='Count', markers=True,
                  color_discrete_sequence=['#f97316'])
    fig.update_layout(height=320)
    return fig

def fig_crime_by_hour(crime_2024):
    hourly = crime_2024.groupby('hour').size().reset_index()
    hourly.columns = ['Hour', 'Count']
    fig = px.bar(hourly, x='Hour', y='Count',
                 color='Count', color_continuous_scale='Oranges')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig

def fig_near_vs_not(crime_2024):
    top_types = crime_2024['CODE_DEFINED'].value_counts().head(6).index
    near      = crime_2024[crime_2024['near_unfit']]['CODE_DEFINED']\
                    .value_counts().reindex(top_types, fill_value=0)
    not_near  = crime_2024[~crime_2024['near_unfit']]['CODE_DEFINED']\
                    .value_counts().reindex(top_types, fill_value=0)
    fig = go.Figure(data=[
        go.Bar(name='Near Unfit (100m)', x=list(top_types),
               y=near.values,     marker_color='#f97316'),
        go.Bar(name='Not Near Unfit',    x=list(top_types),
               y=not_near.values, marker_color='#334155'),
    ])
    fig.update_layout(barmode='group', height=350)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UNFIT PROPERTIES CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def fig_unfit_by_year(unfit):
    yearly = unfit.groupby('year').size().reset_index()
    yearly.columns = ['Year', 'Count']
    fig = px.bar(yearly, x='Year', y='Count',
                 color='Count', color_continuous_scale='Oranges')
    fig.add_hline(y=yearly['Count'].mean(), line_dash='dash', line_color='red',
                  annotation_text=f"Avg: {yearly['Count'].mean():.0f}")
    fig.update_layout(height=380, coloraxis_showscale=False)
    return fig

def fig_open_closed_pie(unfit):
    status = unfit['status_type_name'].value_counts()
    fig = px.pie(values=status.values, names=status.index,
                 color_discrete_sequence=['#ef4444', '#22c55e'], hole=0.45)
    fig.update_layout(height=380)
    return fig

def fig_unfit_by_zip(unfit):
    zips = unfit['zip'].value_counts().head(8).reset_index()
    zips.columns = ['Zip Code', 'Count']
    zips['Zip Code'] = zips['Zip Code'].astype(str)
    fig = px.bar(zips, x='Zip Code', y='Count',
                 color='Count', color_continuous_scale='Reds')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig

def fig_open_by_zip(unfit):
    open_zips = unfit[unfit['status_type_name'] == 'Open']['zip']\
                    .value_counts().head(8).reset_index()
    open_zips.columns = ['Zip Code', 'Open Count']
    open_zips['Zip Code'] = open_zips['Zip Code'].astype(str)
    fig = px.bar(open_zips, x='Zip Code', y='Open Count',
                 color='Open Count', color_continuous_scale='Reds')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FOLIUM MAP
# ══════════════════════════════════════════════════════════════════════════════
def build_map(crime_2024, unfit_clean):
    m = folium.Map(location=[43.048, -76.147], zoom_start=13,
                   tiles='CartoDB positron')
    HeatMap(crime_2024[['LAT', 'LONG']].values.tolist(),
            radius=10, blur=12, min_opacity=0.4).add_to(m)
    for _, row in unfit_clean.iterrows():
        color = 'red' if row.get('status_type_name') == 'Open' else 'gray'
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5, color=color, fill=True, fill_opacity=0.8,
            tooltip=f"{row['address']} | {row.get('status_type_name','?')} | {row['year']}"
        ).add_to(m)
    return m

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def fig_prediction(unfit):
    yearly = unfit.groupby('year').size().reset_index()
    yearly.columns = ['Year', 'Count']
    yearly_fit = yearly[yearly['Year'] <= 2024]

    model = LinearRegression().fit(yearly_fit[['Year']], yearly_fit['Count'])
    future = pd.DataFrame({'Year': [2025, 2026, 2027]})
    preds  = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['Count'],
                         name='Actual', marker_color='#f97316'))
    fig.add_trace(go.Scatter(
        x=future['Year'], y=preds,
        mode='lines+markers+text',
        text=[f"{int(p)}" for p in preds],
        textposition='top center',
        name='Predicted',
        line=dict(color='red', dash='dash'),
        marker=dict(size=10)
    ))
    fig.update_layout(title="Unfit Violations: Actual + Forecast",
                      xaxis_title="Year", yaxis_title="Violations", height=420)
    return fig, future['Year'].tolist(), [int(p) for p in preds]