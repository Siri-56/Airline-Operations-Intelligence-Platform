# =============================================================================
#  03_ML_MODELS.PY
#  Part of Airline Operations Intelligence Platform
#  Handles: Model Training, Evaluation, and ML Charts
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── CONFIGURATION ────────────────────────────────────────────────────────────

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DATA = BASE_DIR / 'data' / 'processed' / 'flights_engineered.parquet'
MODEL_DIR = BASE_DIR / 'outputs' / 'models'
CHART_DIR = BASE_DIR / 'outputs' / 'charts'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
CAT_COLS = ['OP_UNIQUE_CARRIER','ORIGIN','DEST']

def save_chart(fig, name):
    os.makedirs(CHART_DIR, exist_ok=True)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, f'{name}.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'   ✔ {name}.png')

def encode_cats(data, cat_cols):
    d = data.copy()
    for col in cat_cols:
        le = LabelEncoder()
        d[col] = le.fit_transform(d[col].astype(str))
    return d

# ── LOAD DATA ────────────────────────────────────────────────────────────────

print('\n[1/3] Loading engineered data...\n')
if not os.path.exists(PROCESSED_DATA):
    print(f"❌ {PROCESSED_DATA} not found. Run 01_prepare_data.py first.")
    exit()

df = pd.read_parquet(PROCESSED_DATA)

# Sampling for ML
ML_SAMPLE_SIZE = 800_000
if len(df) > ML_SAMPLE_SIZE:
    print(f'   Sampling {ML_SAMPLE_SIZE:,} flights for ML training...')
    df_ml = df.sample(ML_SAMPLE_SIZE, random_state=42).copy()
else:
    df_ml = df.copy()

# ── MODEL 1A: Pre-Departure Classifier ───────────────────────────────────────

print('\n[2/3] Training Model 1A (Pre-Departure)...\n')

FEATURES_1A = CAT_COLS + [
    'MONTH','DAY_OF_WEEK','Departure_Hour',
    'Weekend_Flight','Peak_Hour','Is_Holiday_Period',
    'DISTANCE',
    'Airline_Reliability_Score','Route_Risk_Score',
    'Airport_Congestion_Score','Origin_Delay_Rate',
    'Dest_Delay_Rate','Origin_Avg_Weather_Delay',
    'Route_Avg_Weather_Delay','Busy_Route',
]
if 'Cascade_Delay' in df_ml.columns:
    FEATURES_1A.append('Cascade_Delay')

mdf_1a = encode_cats(df_ml[FEATURES_1A+['DelayRisk']].dropna(), CAT_COLS)
X_1a, y_1a = mdf_1a[FEATURES_1A], mdf_1a['DelayRisk']
X_1a_tr, X_1a_te, y_1a_tr, y_1a_te = train_test_split(
    X_1a, y_1a, test_size=0.2, random_state=42, stratify=y_1a)

rf_1a = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=100,
                                min_samples_split=200, max_features='sqrt', 
                                class_weight='balanced', random_state=42, n_jobs=1)
rf_1a.fit(X_1a_tr, y_1a_tr)

# Save Model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_1a, os.path.join(MODEL_DIR, 'model_1a_rf.joblib'))
print(f"   ✔ Model 1A saved to {MODEL_DIR}")

pred_1a = rf_1a.predict(X_1a_te)
prob_1a = rf_1a.predict_proba(X_1a_te)[:,1]

auc_1a = roc_auc_score(
    y_1a_te,
    prob_1a
)

print("\nMODEL 1A RESULTS")
print(classification_report(
    y_1a_te,
    pred_1a
))

print(f"ROC-AUC : {auc_1a:.4f}")

feature_importance_1A = pd.DataFrame({
    'Feature': FEATURES_1A,
    'Importance': rf_1a.feature_importances_
}).sort_values(
    'Importance',
    ascending=False
)

feature_importance_1A.to_csv(
    MODEL_DIR / 'feature_importance_1A.csv',
    index=False
)
# ── MODEL 1B: Gate-Level Classifier ─────────────────────────────────────────

print('\nTraining Model 1B (Gate-Level)...\n')

FEATURES_1B = FEATURES_1A + [
    'DEP_DELAY',
    'Dep_Delay_Bucket'
]

mdf_1b = encode_cats(
    df_ml[FEATURES_1B + ['DelayRisk']]
    .dropna(),
    CAT_COLS
)

X_1b = mdf_1b[FEATURES_1B]
y_1b = mdf_1b['DelayRisk']

X_1b_tr, X_1b_te, y_1b_tr, y_1b_te = train_test_split(
    X_1b,
    y_1b,
    test_size=0.2,
    random_state=42,
    stratify=y_1b
)

rf_1b = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)

rf_1b.fit(
    X_1b_tr,
    y_1b_tr
)

pred_1b = rf_1b.predict(X_1b_te)

prob_1b = rf_1b.predict_proba(
    X_1b_te
)[:,1]

auc_1b = roc_auc_score(
    y_1b_te,
    prob_1b
)

print(classification_report(
    y_1b_te,
    pred_1b
))

print(f'ROC-AUC : {auc_1b:.4f}')

feature_importance_1B = pd.DataFrame({
    'Feature': FEATURES_1B,
    'Importance': rf_1b.feature_importances_
}).sort_values(
    'Importance',
    ascending=False
)

feature_importance_1B.to_csv(
    MODEL_DIR / 'feature_importance_1B.csv',
    index=False
)

joblib.dump(
    rf_1b,
    MODEL_DIR / 'model_1b_rf.joblib'
)

print("   ✔ Model 1B saved")
# ── MODEL 2B: Regressor ──────────────────────────────────────────────────────

print('\n[3/3] Training Model 2B (Regressor)...\n')

FEATURES_2B = FEATURES_1A + ['DEP_DELAY','Dep_Delay_Bucket']
reg_df = df_ml[(df_ml['ARR_DELAY'] > 15) & (df_ml['ARR_DELAY'] <= 600)].copy()
mdf_2b = encode_cats(reg_df[FEATURES_2B+['ARR_DELAY']].dropna(), CAT_COLS)
X_2b, y_2b = mdf_2b[FEATURES_2B], mdf_2b['ARR_DELAY']
X_2b_tr, X_2b_te, y_2b_tr, y_2b_te = train_test_split(X_2b, y_2b, test_size=0.2, random_state=42)

rfr_2b = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=30,
                                max_features='sqrt', random_state=42, n_jobs=1)
rfr_2b.fit(X_2b_tr, y_2b_tr)

joblib.dump(rfr_2b, os.path.join(MODEL_DIR, 'model_2b_rfr.joblib'))
print(f"   ✔ Model 2B saved to {MODEL_DIR}")

pred_2b = rfr_2b.predict(
    X_2b_te
)

mae_2b = mean_absolute_error(
    y_2b_te,
    pred_2b
)

rmse_2b = np.sqrt(
    mean_squared_error(
        y_2b_te,
        pred_2b
    )
)

r2_2b = r2_score(
    y_2b_te,
    pred_2b
)

print("\nMODEL 2B RESULTS")

print(f"MAE  : {mae_2b:.2f}")
print(f"RMSE : {rmse_2b:.2f}")
print(f"R²   : {r2_2b:.4f}")

feature_importance_2B = pd.DataFrame({
    'Feature': FEATURES_2B,
    'Importance': rfr_2b.feature_importances_
}).sort_values(
    'Importance',
    ascending=False
)

feature_importance_2B.to_csv(
    MODEL_DIR / 'feature_importance_2B.csv',
    index=False
)

print('\n✅ ML modeling complete.')

fig, ax = plt.subplots(
    figsize=(10,6)
)

feature_importance_1B.head(20).sort_values(
    'Importance'
).plot(
    x='Feature',
    y='Importance',
    kind='barh',
    ax=ax
)

ax.set_title(
    'Top 20 Feature Importance (Model 1B)'
)

save_chart(
    fig,
    'chart16!_feature_importance_all'
)

