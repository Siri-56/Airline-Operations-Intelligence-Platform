# =============================================================================
#  04_POWERBI_EXPORTS.PY
#  Part of Airline Operations Intelligence Platform
#  Handles: CSV Exporting for Power BI and Strategic Summary
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ── CONFIGURATION ────────────────────────────────────────────────────────────

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DATA = BASE_DIR / "data" / "processed" / "flights_engineered.parquet"
EXPORT_DIR = BASE_DIR / "outputs" / "exports"

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ── LOAD DATA ────────────────────────────────────────────────────────────────

print('\n[1/2] Loading engineered data...\n')
if not os.path.exists(PROCESSED_DATA):
    print(f"❌ {PROCESSED_DATA} not found. Run 01_prepare_data.py first.")
    exit()

df = pd.read_parquet(PROCESSED_DATA)

# ── EXPORT CSVS ──────────────────────────────────────────────────────────────

print('\n[2/2] Exporting Power BI datasets...\n')
os.makedirs(EXPORT_DIR, exist_ok=True)

EXPORT_COLS = [
    'FL_DATE','YEAR','MONTH','DAY_OF_WEEK','Airline_Name','OP_UNIQUE_CARRIER',
    'ORIGIN','DEST','Route','Season',
    'CRS_DEP_TIME','Departure_Hour','Weekend_Flight','Peak_Hour',
    'DEP_DELAY','ARR_DELAY','DISTANCE',
    'CANCELLED','DelayRisk','Delay_Cost_USD',
]
if 'TAIL_NUM' in df.columns:
    EXPORT_COLS.append('TAIL_NUM')
if 'Cascade_Delay' in df.columns:
    EXPORT_COLS.append('Cascade_Delay')

# Export sampled main dataset
export_main = df[[c for c in EXPORT_COLS if c in df.columns]].sample(
    n=min(500_000,len(df)), random_state=42)
export_main.to_csv(os.path.join(EXPORT_DIR, 'flights_clean_BTS.csv'), index=False)
print(f'   ✔ flights_clean_BTS.csv          ({len(export_main):,} rows)')

# Airline Performance
airline_export = df.groupby(['OP_UNIQUE_CARRIER','Airline_Name']).agg(
    Total_Flights=('DelayRisk','count'),
    Delayed_Flights=('DelayRisk','sum'),
    Delay_Rate_Pct=('DelayRisk',lambda x: round(x.mean()*100,2)),
    Avg_Arrival_Delay=('ARR_DELAY',lambda x: round(x.mean(),2)),
    Total_Delay_Cost_USD=('Delay_Cost_USD','sum'),
).reset_index().round(2)
airline_export.to_csv(os.path.join(EXPORT_DIR, 'airline_performance_BTS.csv'), index=False)
print(f'   ✔ airline_performance_BTS.csv')

# KPI SUMMARY

kpi_summary = pd.DataFrame({

    'Metric': [

        'Total Flights',
        'Delay Rate %',
        'Average Arrival Delay',
        'Total Delay Cost USD',
        'Total Airlines',
        'Total Airports',
        'Total Routes'

    ],

    'Value': [

        len(df),

        round(
            df['DelayRisk'].mean() * 100,
            2
        ),

        round(
            df['ARR_DELAY'].mean(),
            2
        ),

        round(
            df['Delay_Cost_USD'].sum(),
            2
        ),

        df['Airline_Name'].nunique(),

        df['ORIGIN'].nunique(),

        df['Route'].nunique()

    ]
})

kpi_summary.to_csv(
    EXPORT_DIR / 'kpi_summary_BTS.csv',
    index=False
)

print(
    '   ✔ kpi_summary_BTS.csv'
)

airport_export = (
    df.groupby('ORIGIN')
    .agg(
        Total_Flights=('DelayRisk','count'),
        Delay_Rate=('DelayRisk','mean'),
        Avg_Dep_Delay=('DEP_DELAY','mean'),
        Avg_Arr_Delay=('ARR_DELAY','mean')
    )
    .reset_index()
)

airport_export.to_csv(
    EXPORT_DIR / 'airport_performance_BTS.csv',
    index=False
)

print(
    '   ✔ airport_performance_BTS.csv'
)

route_export = (
    df.groupby('Route')
    .agg(
        Total_Flights=('DelayRisk','count'),
        Delay_Rate=('DelayRisk','mean'),
        Avg_Arr_Delay=('ARR_DELAY','mean'),
        Route_Risk=('Route_Risk_Score','mean')
    )
    .reset_index()
)

route_export.to_csv(
    EXPORT_DIR / 'route_performance_BTS.csv',
    index=False
)

print(
    '   ✔ route_performance_BTS.csv'
)

monthly_export = (
    df.groupby('MONTH')
    .agg(
        Flights=('DelayRisk','count'),
        Delay_Rate=('DelayRisk','mean'),
        Avg_Delay=('ARR_DELAY','mean'),
        Delay_Cost=('Delay_Cost_USD','sum')
    )
    .reset_index()
)

monthly_export.to_csv(
    EXPORT_DIR / 'monthly_summary_BTS.csv',
    index=False
)

print(
    '   ✔ monthly_summary_BTS.csv'
)

if 'Cascade_Delay' in df.columns:

    propagation_summary = (
        df.groupby('Airline_Name')
        .agg(
            Flights=('DelayRisk','count'),
            Cascade_Rate=('Cascade_Delay','mean')
        )
        .reset_index()
    )

    propagation_summary.to_csv(
        EXPORT_DIR / 'propagation_summary_BTS.csv',
        index=False
    )

    print(
        '   ✔ propagation_summary_BTS.csv'
    )

    propagation_routes = (
        df[df['Cascade_Delay'] == 1]
        .groupby('Route')
        .size()
        .reset_index(name='Cascade_Count')
        .sort_values(
            'Cascade_Count',
            ascending=False
        )
    )

    propagation_routes.to_csv(
        EXPORT_DIR / 'propagation_routes_BTS.csv',
        index=False
    )

    print(
        '   ✔ propagation_routes_BTS.csv'
    )
 
print('\n══════════════════════════════════════')
print(' AIRLINE OPERATIONS SUMMARY')
print('══════════════════════════════════════')

print(
    f'Total Flights : {len(df):,}'
)

print(
    f'Delay Rate    : {df["DelayRisk"].mean()*100:.1f}%'
)

print(
    f'Avg Delay     : {df["ARR_DELAY"].mean():.1f} min'
)

print(
    f'Total Cost    : ${df["Delay_Cost_USD"].sum():,.0f}'
)

if 'Cascade_Delay' in df.columns:

    print(
        f'Cascade Rate  : '
        f'{df["Cascade_Delay"].mean()*100:.1f}%'
    )

print('\n✅ Power BI exports complete.')
