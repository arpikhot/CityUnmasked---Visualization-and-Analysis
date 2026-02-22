#  CityUnmasked — Visualization and Analysis

---

##  Project Overview

CityUnmasked investigates the relationship between **urban decay** (unfit properties, vacant properties, code violations) and **crime patterns** in Syracuse, NY using four real municipal datasets. The project identifies where blight and crime co-occur, measures the strength of that co-occurrence, classifies every zip code by decay type, and predicts where crime will concentrate next.

**Core Thesis:**
> *We are NOT claiming crime creates blight or blight creates crime. We ARE showing that when they find each other in the same geography, they don't let go — and Syracuse's own data shows us exactly where that grip is tightening right now.*

**Core Question:**
> *Do neighborhoods with more unfit and vacant properties experience disproportionately higher crime rates — and can we predict where intervention is most needed?*

---

---

##  Project Structure

```
CityUnmasked/
├── dashboard.py                    ← Main Streamlit app (run this)
│
├── analysis/                       ← Data processing and chart functions
│   ├── __init__.py
│   ├── crime.py                    ← Crime data loading and charts
│   ├── unfit.py                    ← Unfit properties loading and charts
│   ├── vacant.py                   ← Vacant properties loading and charts
│   ├── code_violations.py          ← Code violations loading, filtering, tiering
│   ├── decay_index.py              ← Spatial join, A/B/C classification, Urban Decay Index
│   ├── models.py                   ← Granger causality, Random Forest
│   ├── map_builder.py              ← Folium map construction
│   └── crime_risk_dev.py           ← Multi-year crime hotspot prediction model ← NEW
│
├── tabs/                           ← Dashboard tab rendering
│   ├── __init__.py
│   ├── tab_crime.py
│   ├── tab_unfit.py
│   ├── tab_vacant.py
│   ├── tab_decay_index.py
│   ├── tab_code_violations.py
│   ├── tab_map.py
│   └── tab_prediction.py          ← Updated with hotspot model UI ← UPDATED
│
├── crime_clean.csv                 ← 25,752 crime incidents (2023–2025)
├── Unfit_Properties.csv            ← 264 unfit violations (2014–2025)
├── Vacant_Properties.csv           ← 1,651 vacant registrations
├── code_violations.csv             ← 140,726 code violations (2017–2026)
├── requirements.txt
└── README.md
```

---

---

## Multi-Year Crime Hotspot Prediction

### What It Does

The hotspot prediction model uses **2023–2025 crime data** to identify chronic high-risk grid cells across the city and predict which areas are most likely to become Q4 crime clusters.


---

##  Dashboard — 7 Tabs

| Tab | Contents |
|---|---|
| 📊 Crime Analysis | Top crime types, serious vs QoL split, monthly patterns, hourly distribution |
| 🏚️ Unfit Properties | Annual violation trend, open/closed rate, zip code concentration |
| 🏘️ Vacant Properties | Neighborhood breakdown, active vs resolved, zip distribution |
| 📉 Urban Decay Index | A/B/C classification, scatter quadrant, risk ranking, economic abandonment zones, Granger causality |
| ⚠️ Code Violations | Tier breakdown by year, violation geography, bidirectional Granger test (108 months) |
| 🗺️ Map | Three-layer Folium: crime heatmap + unfit markers + vacant density. Layer toggles. |
| 🔮 Prediction | Hotspot risk heatmap (2023–2025), Top 10 chronic risk grids, logistic regression explanation, policy recommendations |

---

##  Datasets

| Dataset | File | Records | Key Columns |
|---|---|---|---|
| Crime | `crime_clean.csv` | 25,752 | LAT, LON, CRIME_TYPE, SEVERITY, SEASON, TIME_OF_DAY, YEAR, MONTH |
| Unfit Properties | `Unfit_Properties.csv` | 264 | Latitude, Longitude, status_type_name, violation_date, zip |
| Vacant Properties | `Vacant_Properties.csv` | 1,651 | Latitude, Longitude, neighborhood, Zip, VPR_valid |
| Code Violations | `code_violations.csv` | 140,726 (92,790 filtered) | Latitude, Longitude, complaint_type_name, violation, violation_date, Neighborhood |

---

##  Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
python -m streamlit run dashboard.py
```

Opens at `http://localhost:8501`

---

##  Dependencies

```
streamlit>=1.32.0
streamlit-folium>=0.18.0
folium>=0.16.0
plotly>=5.20.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
statsmodels>=0.14.0
scipy>=1.12.0
geopy>=2.4.0
```

```bash
pip install -r requirements.txt
```

---


---




---

