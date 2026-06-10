# =============================================================================
#  01_PREPARE_DATA.PY
#  Part of Airline Operations Intelligence Platform
#  Handles: Data Loading, Cleaning, Feature Engineering, Propagation Analysis
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from pathlib import Path


warnings.filterwarnings('ignore')

# ── CONFIGURATION ────────────────────────────────────────────────────────────

AIRLINE_NAMES = {
    'AA': 'American Airlines',  'AS': 'Alaska Airlines',
    'B6': 'JetBlue',            'DL': 'Delta Air Lines',
    'F9': 'Frontier Airlines',  'G4': 'Allegiant Air',
    'HA': 'Hawaiian Airlines',  'MQ': 'Envoy Air',
    'NK': 'Spirit Airlines',    'OH': 'PSA Airlines',
    'OO': 'SkyWest Airlines',   'UA': 'United Airlines',
    'WN': 'Southwest Airlines', 'YX': 'Republic Airways',
}

CANCEL_CODES = {'A': 'Carrier', 'B': 'Weather', 'C': 'NAS', 'D': 'Security'}

COST_PER_DELAY_MIN    = 74.24
COST_PER_CANCELLATION = 8_000

# Updated paths for the new structure

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_FILES = [
    DATA_DIR / 'Jan2025.csv',
    DATA_DIR / 'Feb2025.csv',
    DATA_DIR / 'Mar2025.csv',
    DATA_DIR / 'Apr2025.csv',
    DATA_DIR / 'May2025.csv',
    DATA_DIR / 'Jun2025.csv',
]

USECOLS = [
    'YEAR', 'MONTH', 'DAY_OF_WEEK', 'FL_DATE',
    'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST',
    'TAIL_NUM',
    'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY',
    'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY',
    'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED',
    'AIR_TIME', 'ACTUAL_ELAPSED_TIME', 'DISTANCE',
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY',
    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
]

DTYPES = {
    'YEAR': 'int16', 'MONTH': 'int8', 'DAY_OF_WEEK': 'int8',
    'OP_UNIQUE_CARRIER': 'category',
    'ORIGIN': 'category', 'DEST': 'category',
    'TAIL_NUM': 'str',
    'CRS_DEP_TIME': 'int16', 'CANCELLED': 'float32', 'DIVERTED': 'float32',
    'DEP_DELAY': 'float32', 'ARR_DELAY': 'float32',
    'AIR_TIME': 'float32', 'ACTUAL_ELAPSED_TIME': 'float32',
    'DISTANCE': 'float32',
    'CARRIER_DELAY': 'float32', 'WEATHER_DELAY': 'float32',
    'NAS_DELAY': 'float32', 'SECURITY_DELAY': 'float32',
    'LATE_AIRCRAFT_DELAY': 'float32',
}

# ── LOAD & COMBINE ───────────────────────────────────────────────────────────

print('\n[1/4] Loading BTS flight data...\n')

def safe_read(filepath, usecols, dtypes):
    sample = pd.read_csv(filepath, nrows=1)
    available = [c for c in usecols if c in sample.columns]
    safe_dtypes = {k: v for k, v in dtypes.items() if k in available}
    return pd.read_csv(
        filepath, usecols=available, dtype=safe_dtypes,
        parse_dates=['FL_DATE'],
        date_format='%m/%d/%Y %I:%M:%S %p'
    )

chunks = []
month_names = ['January','February','March','April','May','June']

for i, f in enumerate(DATA_FILES):
    try:
        tmp = safe_read(f, USECOLS, DTYPES)
        chunks.append(tmp)
        print(f'   ✔ {month_names[i]:10s} 2025 : {len(tmp):>8,} flights')
    except FileNotFoundError:
        print(f'   ✗ {f} not found')

if not chunks:
    print("❌ No data files found exiting.")
    exit()

df = pd.concat(chunks, ignore_index=True)
del chunks

HAS_TAIL = 'TAIL_NUM' in df.columns
print(f'\n   Total flights loaded : {len(df):>10,}')

# ── DATA CLEANING ────────────────────────────────────────────────────────────

print('\n[2/4] Cleaning data...\n')

df_cancelled = df[df['CANCELLED'] == 1.0].copy()
df_cancelled['CANCELLATION_REASON'] = (
    df_cancelled['CANCELLATION_CODE'].map(CANCEL_CODES).fillna('Unknown')
)
df_cancelled['Airline_Name'] = (
    df_cancelled['OP_UNIQUE_CARRIER']
    .map(AIRLINE_NAMES)
    .fillna(df_cancelled['OP_UNIQUE_CARRIER'].astype(str))
)

df = df[(df['CANCELLED'] == 0.0) & (df['DIVERTED'] == 0.0)].copy()
df.dropna(subset=['ARR_DELAY'], inplace=True)

delay_cause_cols = ['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY',
                    'SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
df[delay_cause_cols] = df[delay_cause_cols].fillna(0)
df['DEP_DELAY'].fillna(df['DEP_DELAY'].median(), inplace=True)

if HAS_TAIL:
    df['TAIL_NUM'] = df['TAIL_NUM'].fillna('UNKNOWN').astype(str).str.strip()

df['Airline_Name'] = (
    df['OP_UNIQUE_CARRIER']
    .map(AIRLINE_NAMES)
    .fillna(df['OP_UNIQUE_CARRIER'].astype(str))
)

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

print('\n[3/4] Engineering features...\n')

df['Departure_Hour'] = df['CRS_DEP_TIME'] // 100
df['Weekend_Flight'] = df['DAY_OF_WEEK'].isin([6,7]).astype('int8')
df['Peak_Hour']      = df['Departure_Hour'].isin([7,8,9,16,17,18,19]).astype('int8')
df['Season']         = df['MONTH'].map({
    1:'Winter',2:'Winter',3:'Spring',
    4:'Spring',5:'Spring',6:'Summer'
})
df['DelayRisk'] = (df['ARR_DELAY'] > 15).astype('int8')
df['Route']     = df['ORIGIN'].astype(str) + '-' + df['DEST'].astype(str)

# Airline Reliability Score
airline_delay_rate = df.groupby('OP_UNIQUE_CARRIER')['DelayRisk'].mean()
df['Airline_Reliability_Score'] = (
    df['OP_UNIQUE_CARRIER'].astype(str).map(1 - airline_delay_rate) * 100
).round(1)

# Route Risk Score
route_stats = df.groupby('Route').agg(
    route_delay_rate=('DelayRisk','mean'),
    route_avg_delay=('ARR_DELAY','mean')
)
route_stats['Route_Risk_Score'] = (
    route_stats['route_delay_rate'] * route_stats['route_avg_delay']
).round(2)
df = df.merge(route_stats[['Route_Risk_Score']], on='Route', how='left')

# Airport Congestion Score
airport_congestion = df.groupby('ORIGIN')['DEP_DELAY'].mean().round(2)
df['Airport_Congestion_Score'] = df['ORIGIN'].map(airport_congestion)

# Historical rates
origin_delay_rate = df.groupby('ORIGIN')['DelayRisk'].mean()
dest_delay_rate   = df.groupby('DEST')['DelayRisk'].mean()
df['Origin_Delay_Rate'] = df['ORIGIN'].astype(str).map(origin_delay_rate).fillna(0)
df['Dest_Delay_Rate']   = df['DEST'].astype(str).map(dest_delay_rate).fillna(0)

origin_avg_dep     = df.groupby('ORIGIN')['DEP_DELAY'].mean()
origin_avg_weather = df.groupby('ORIGIN')['WEATHER_DELAY'].mean()
route_avg_weather  = df.groupby('Route')['WEATHER_DELAY'].mean()
route_avg_dep      = df.groupby('Route')['DEP_DELAY'].mean()
df['Origin_Avg_Dep_Delay']     = df['ORIGIN'].astype(str).map(origin_avg_dep).fillna(0)
df['Origin_Avg_Weather_Delay'] = df['ORIGIN'].astype(str).map(origin_avg_weather).fillna(0)
df['Route_Avg_Weather_Delay']  = df['Route'].map(route_avg_weather).fillna(0)
df['Route_Avg_Dep_Delay']      = df['Route'].map(route_avg_dep).fillna(0)

route_counts = df['Route'].value_counts()
df['Busy_Route'] = (df['Route'].map(route_counts).fillna(0) > 500).astype('int8')

holiday_dates = pd.to_datetime([
    '2025-01-01','2025-01-02','2025-01-03','2025-01-04','2025-01-05',
    '2025-03-15','2025-03-16','2025-03-17','2025-03-18','2025-03-19',
    '2025-03-20','2025-03-21','2025-03-22','2025-03-23','2025-03-24','2025-03-25',
    '2025-05-23','2025-05-24','2025-05-25','2025-05-26','2025-05-27',
    '2025-06-18','2025-06-19','2025-06-20','2025-06-21',
])
df['Is_Holiday_Period'] = df['FL_DATE'].isin(holiday_dates).astype('int8')

df['Dep_Delay_Bucket'] = pd.cut(
    df['DEP_DELAY'],
    bins=[-np.inf,-5,0,5,15,30,60,np.inf],
    labels=[0,1,2,3,4,5,6]
).astype(float).fillna(2)

df['Positive_Arr_Delay'] = df['ARR_DELAY'].clip(lower=0)
df['Delay_Cost_USD']     = (df['Positive_Arr_Delay'] * COST_PER_DELAY_MIN).round(2)

# ── DELAY PROPAGATION ANALYSIS ───────────────────────────────────────────────

propagation_summary = None

if HAS_TAIL:

    print('\n[4/4] Delay Propagation Analysis (TAIL_NUM)...\n')

    prop_df = df[
        df['TAIL_NUM'] != 'UNKNOWN'
    ].copy()

    def hhmm_to_minutes(series):
        return (series // 100) * 60 + (series % 100)

    prop_df['Dep_Minutes'] = hhmm_to_minutes(
        prop_df['CRS_DEP_TIME'].astype(int)
    )

    prop_df = prop_df.sort_values(
        ['TAIL_NUM', 'FL_DATE', 'Dep_Minutes']
    )

    prop_df['Prev_ARR_DELAY'] = (
        prop_df.groupby('TAIL_NUM')['ARR_DELAY']
        .shift(1)
    )

    prop_df['Prev_DEST'] = (
        prop_df.groupby('TAIL_NUM')['DEST']
        .shift(1)
    )

    prop_df['Prev_FL_DATE'] = (
        prop_df.groupby('TAIL_NUM')['FL_DATE']
        .shift(1)
    )

    prop_df['Same_Day'] = (
        prop_df['FL_DATE']
        ==
        prop_df['Prev_FL_DATE']
    )

    prop_df_consec = prop_df[
        prop_df['Same_Day']
        &
        prop_df['Prev_ARR_DELAY'].notna()
        &
        (
            prop_df['Prev_DEST'].astype(str)
            ==
            prop_df['ORIGIN'].astype(str)
        )
    ].copy()

    prop_df_consec['Prev_Was_Late'] = (
        prop_df_consec['Prev_ARR_DELAY'] > 15
    ).astype(int)

    prop_df_consec['This_Dep_Late'] = (
        prop_df_consec['DEP_DELAY'] > 15
    ).astype(int)

    prop_df_consec['Cascade_Flag'] = (
        (
            prop_df_consec['Prev_Was_Late'] == 1
        )
        &
        (
            prop_df_consec['This_Dep_Late'] == 1
        )
    ).astype(int)

    cascade_index = (
        prop_df_consec[
            prop_df_consec['Cascade_Flag'] == 1
        ].index
    )

    df['Cascade_Delay'] = 0

    df.loc[
        df.index.isin(cascade_index),
        'Cascade_Delay'
    ] = 1

    propagation_summary = prop_df_consec.copy()

    print(
        f'   Consecutive flights analysed : {len(prop_df_consec):,}'
    )

    print(
        f'   Cascade delays detected      : '
        f'{prop_df_consec["Cascade_Flag"].sum():,}'
    )

    print('✅ Delay Propagation complete.')


# ── SAVE PROCESSED DATA ───────────────────────────────────────────────────────

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Main engineered dataset
flights_path = os.path.join(
    PROCESSED_DIR,
    'flights_engineered.parquet'
)

df.to_parquet(
    flights_path,
    index=False
)

# Propagation dataset
if propagation_summary is not None:

    propagation_path = os.path.join(
        PROCESSED_DIR,
        'propagation_analysis.parquet'
    )

    propagation_summary.to_parquet(
        propagation_path,
        index=False
    )

# Summary
print('\n✅ Saved Files:')

print(f'   {flights_path}')

if propagation_summary is not None:
    print(f'   {propagation_path}')

print(f'   {top_airports_path}')

print(f'\nDataset Shape : {df.shape}')
print(f'Total Columns : {len(df.columns)}')