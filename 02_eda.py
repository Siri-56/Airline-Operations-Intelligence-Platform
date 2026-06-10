# =============================================================================
#  02_EDA.PY
#  Part of Airline Operations Intelligence Platform
#  Handles: Exploratory Data Analysis and Chart Generation
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── CONFIGURATION ────────────────────────────────────────────────────────────

PALETTE = ['#2C3E50', '#E74C3C', '#27AE60', '#F39C12', '#2980B9',
           '#8E44AD', '#16A085', '#D35400', '#C0392B', '#1ABC9C']
sns.set_theme(style='whitegrid', palette=PALETTE)
plt.rcParams.update({'figure.dpi': 130, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'figure.facecolor': 'white'})



BASE_DIR = Path(__file__).resolve().parent

CHART_DIR = BASE_DIR / "outputs" / "charts"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "flights_engineered.parquet"

def save_chart(fig, name):

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.tight_layout()

    path = CHART_DIR / f"{name}.png"

    fig.savefig(
        path,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(f'   ✔ {name}.png')

# ── LOAD DATA ────────────────────────────────────────────────────────────────

print('\n[1/2] Loading engineered data...\n')
if not os.path.exists(PROCESSED_DATA):
    print(f"❌ {PROCESSED_DATA} not found. Run 01_prepare_data.py first.")
    exit()

df = pd.read_parquet(PROCESSED_DATA)
print(f"Dataset Shape : {df.shape}")
print(f"Columns       : {len(df.columns)}")
MONTH_LABELS = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun'}
delay_cause_cols = ['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY',
                    'SECURITY_DELAY','LATE_AIRCRAFT_DELAY']

# ── GENERATE CHARTS ──────────────────────────────────────────────────────────

print('\n[2/2] Generating EDA charts...\n')

# Chart 1: Monthly Delay Rate
fig, ax1 = plt.subplots(figsize=(12,5))
monthly_dr = df.groupby('MONTH')['DelayRisk'].mean() * 100
x = [MONTH_LABELS[m] for m in monthly_dr.index]
ax1.bar(x, monthly_dr.values, color=PALETTE[1], alpha=0.7, label='Delay Rate %')
ax1.set_ylabel('Delay Rate (%)', color=PALETTE[1])
ax1.set_title('Monthly Delay Rate', fontweight='bold')
save_chart(fig, 'chart1_monthly_delay_rate')

# Chart 2: Delay Causes Breakdown
cause_totals = df[delay_cause_cols].sum().sort_values(ascending=True)
cause_labels = ['Late Aircraft','Security','NAS/ATC','Weather','Carrier']
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.barh(cause_labels, cause_totals.values/1e6, color=PALETTE[:5])
ax.set_title('Total Delay Minutes by Cause', fontweight='bold')
ax.set_xlabel('Total Delay Minutes (Millions)')
for bar, val in zip(bars, cause_totals.values/1e6):
    ax.text(val+0.1, bar.get_y()+bar.get_height()/2,
            f'{val:.1f}M', va='center', fontweight='bold')
save_chart(fig, 'chart2!_delay_causes_breakdown')

# Chart 3: Airline Reliability Score
airline_perf = df.groupby('Airline_Name').agg(
    Reliability=('Airline_Reliability_Score','mean')
).reset_index().sort_values('Reliability', ascending=True)

fig, ax = plt.subplots(figsize=(10,6))
colors = [PALETTE[2] if r>=70 else PALETTE[3] if r>=55 else PALETTE[1]
          for r in airline_perf['Reliability']]
bars = ax.barh(airline_perf['Airline_Name'], airline_perf['Reliability'],
               color=colors, edgecolor='white')
ax.set_title('Airline Reliability Score', fontweight='bold')
ax.set_xlabel('Reliability Score (0–100)')
save_chart(fig, 'chart3!_airline_reliability')

# Chart 4: Airport Congestion
top_airports = (
    df.groupby('ORIGIN')
    .agg(Avg_Dep_Delay=('DEP_DELAY','mean'), Flights=('DEP_DELAY','count'))
    .query('Flights >= 5000')
    .nlargest(15,'Avg_Dep_Delay')
    .reset_index()
)
fig, ax = plt.subplots(figsize=(10,6))
ax.barh(top_airports['ORIGIN'], top_airports['Avg_Dep_Delay'], color=PALETTE[1])
ax.set_title('Top 15 Most Congested Airports', fontweight='bold')
ax.invert_yaxis()
save_chart(fig, 'chart4!_airport_congestion')


# Chart 5: Top Risk Routes

route_risk = (
    df.groupby('Route')
    .agg(
        Risk=('Route_Risk_Score', 'mean'),
        Flights=('Route', 'count')
    )
    .query('Flights >= 200')
    .nlargest(15, 'Risk')
    .reset_index()
)

fig, ax = plt.subplots(figsize=(11,6))

ax.barh(
    route_risk['Route'],
    route_risk['Risk'],
    color=PALETTE[3]
)

ax.set_title(
    'Top 15 Highest Risk Routes',
    fontweight='bold'
)

ax.invert_yaxis()

save_chart(
    fig,
    'chart5_top_risk_routes'
)

# Chart 6: Delay Distribution

fig, ax = plt.subplots(figsize=(10,5))

sns.histplot(
    df['ARR_DELAY'].clip(-50,300),
    bins=60,
    kde=True,
    ax=ax,
    color=PALETTE[1]
)

ax.set_title(
    'Arrival Delay Distribution',
    fontweight='bold'
)

save_chart(
    fig,
    'chart6_delay_distribution'
)

# Chart 7: Delay by Hour

hourly = (
    df.groupby('Departure_Hour')
    ['ARR_DELAY']
    .mean()
)

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    hourly.index,
    hourly.values,
    marker='o'
)

ax.set_title(
    'Average Delay by Departure Hour',
    fontweight='bold'
)

ax.set_xlabel('Hour')
ax.set_ylabel('Avg Delay (min)')

save_chart(
    fig,
    'chart7_delay_by_hour'
)

# Chart 9: Distance vs Delay

sample_df = df.sample(
    min(50000, len(df)),
    random_state=42
)
fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=sample_df,
    x='DISTANCE',
    y='ARR_DELAY',
    alpha=0.2,
    ax=ax
)

ax.set_title(
    'Distance vs Arrival Delay',
    fontweight='bold'
)

save_chart(
    fig,
    'chart9_distance_vs_delay'
)

# Chart 10 — Airline Delay Cause Heatmap

heatmap_df = (
    df.groupby('Airline_Name')[
        [
            'CARRIER_DELAY',
            'WEATHER_DELAY',
            'NAS_DELAY',
            'SECURITY_DELAY',
            'LATE_AIRCRAFT_DELAY'
        ]
    ]
    .mean()
)

fig, ax = plt.subplots(figsize=(12,6))

sns.heatmap(
    heatmap_df,
    cmap='Reds',
    ax=ax
)

ax.set_title(
    'Average Delay Cause by Airline',
    fontweight='bold'
)

save_chart(
    fig,
    'chart10_airline_cause_heatmap'
)

# Chart 11: Financial Impact by Airline

financial = (
    df.groupby('Airline_Name')
    ['Delay_Cost_USD']
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

fig, ax = plt.subplots(figsize=(10,6))

financial.plot(
    kind='bar',
    ax=ax
)

ax.set_title(
    'Financial Impact by Airline',
    fontweight='bold'
)

ax.yaxis.set_major_formatter(
    mticker.StrMethodFormatter('${x:,.0f}')
)

save_chart(
    fig,
    'chart11_financial_impact_airline'
)

# Chart 12: Day of Week Delay Rate

dow = (
    df.groupby('DAY_OF_WEEK')
    ['DelayRisk']
    .mean()
    * 100
)

fig, ax = plt.subplots(figsize=(8,5))

dow.plot(
    kind='bar',
    ax=ax
)

ax.set_title(
    'Delay Rate by Day of Week',
    fontweight='bold'
)

ax.set_ylabel('Delay Rate %')

save_chart(
    fig,
    'chart12_day_of_week_delay'
)
# Chart 13: Cascade Delay Rate by Airline
if 'Cascade_Delay' in df.columns:

    cascade = (
        df.groupby('Airline_Name')
        ['Cascade_Delay']
        .mean()
        .sort_values(ascending=False)
        .head(15)
        * 100
    )

    fig, ax = plt.subplots(figsize=(10,6))

    cascade.plot(
        kind='barh',
        ax=ax
    )

    ax.set_title(
        'Cascade Delay Rate by Airline',
        fontweight='bold'
    )

    save_chart(
        fig,
        'chart13_cascade_rate_airline'
    )
 # Chart 14 — Propagation Summary

if 'Cascade_Delay' in df.columns:

    counts = df['Cascade_Delay'].value_counts()

    fig, ax = plt.subplots(figsize=(6,6))

    ax.pie(
        counts,
        labels=['No Cascade', 'Cascade'],
        autopct='%1.1f%%'
    )

    ax.set_title(
        'Delay Propagation Analysis',
        fontweight='bold'
    )

    save_chart(
        fig,
        'chart14!_propagation_analysis'
    )
# Chart 15 — Top Cascade Routes

if 'Cascade_Delay' in df.columns:

    routes = (
        df[df['Cascade_Delay'] == 1]
        ['Route']
        .value_counts()
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10,6))

    routes.plot(
        kind='barh',
        ax=ax
    )

    ax.set_title(
        'Routes with Highest Cascade Delays',
        fontweight='bold'
    )

    save_chart(
        fig,
        'chart15_cascade_routes'
    )
print('\n✅ EDA charts complete.')
