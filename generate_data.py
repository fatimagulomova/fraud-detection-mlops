import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from scipy import stats

# ==================================================================================PATHS================================================================================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MONTHLY_DIR = os.path.join(DATA_DIR, "monthly")

# ==================================================================================LOAD DATA================================================================================================================================

os.makedirs(MONTHLY_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "fraud_dataset.csv"))

with open(os.path.join(DATA_DIR, "data_params.json"), "r") as file:
    data_params = json.load(file)

# ===========================================================================SYNTHETIC DATA==============================================================================

metadata = Metadata.load_from_dict(metadata_dict=data_params)
synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
synthesizer.fit(df)

synthetic_df = synthesizer.sample(num_rows=len(df))
synthetic_df.to_csv("data/fraud_synthetic.csv", index=False)

# ===========================================================================DRIFT GENERATION======================================================================

def generate_drifted_month(base_df, month: int):
    """Realistic per-column drift strengths."""
    drifted = base_df.copy()

    # Tuned drift per column — risk scores drift much more slowly
    drift_config = {
        'amount':            0.03,   # moderate financial drift
        'hour':              0.003,   # minimal — time patterns stable
        'device_risk_score': 0.008,  # slow — was 0.05, way too fast
        'ip_risk_score':     0.008,  # slow — was too fast
    }

    for col, strength in drift_config.items():
        noise = np.random.normal(
            loc=strength * month,
            scale=base_df[col].std() * strength,
            size=len(drifted)
        )
        drifted[col] = (drifted[col] + noise).clip(lower=0)

    drifted['device_risk_score'] = drifted['device_risk_score'].clip(0, 1)
    drifted['ip_risk_score'] = drifted['ip_risk_score'].clip(0, 1)
    drifted['month'] = month
    drifted['hour'] = drifted['hour'].clip(0, 23).round()

    return drifted


monthly_data = {}

for m in range(1, 13):

    monthly_data[m] = generate_drifted_month(
        synthetic_df,
        month=m
    )

    file_path = os.path.join(MONTHLY_DIR, f"month_{m:02d}.csv")
    monthly_data[m].to_csv(file_path, index=False)

    print(
        f"Month {m:02d} | "
        f"amount={monthly_data[m]['amount'].mean():.2f} | "
        f"device_risk={monthly_data[m]['device_risk_score'].mean():.4f} | "
        f"ip_risk={monthly_data[m]['ip_risk_score'].mean():.4f}"
    )

# ====================================================================DRIFT DETECTION===================================================================================

def detect_drift(baseline_df, new_df, threshold=0.05):

    numeric_cols = [
        'amount',
        'hour',
        'device_risk_score',
        'ip_risk_score'
    ]

    drift_found = False

    for col in numeric_cols:

        stat, p_value = stats.ks_2samp(
            baseline_df[col],
            new_df[col]
        )

        status = (
            "🔴 DRIFT"
            if p_value < threshold
            else "🟢 OK"
        )

        print(
            f"{col:22s} "
            f"KS={stat:.4f} "
            f"p={p_value:.4f} "
            f"{status}"
        )

        if p_value < threshold:
            drift_found = True

    return drift_found

# ========================================================TEST DRIFT=========================================================================================================

month6 = pd.read_csv(
    os.path.join(MONTHLY_DIR, "month_06.csv")
)

print("\n=== Drift report: Month 6 vs Baseline ===")

trigger = detect_drift(
    synthetic_df,
    month6
)