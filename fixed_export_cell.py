# ═══════════════════════════════════════════════════════════════════
# UPDATED EXPORT CELL (FIXED)
# Includes ablation + cross-dataset only
# NO statistical significance (we skipped that)
# ═══════════════════════════════════════════════════════════════════

import pickle
import os
import shutil
from google.colab import files

export_dir = '/content/export_to_vscode'
os.makedirs(export_dir, exist_ok=True)

print("\nExporting all models + paper results for Flask API...")

# 1. Best model
with open(f'{export_dir}/fraud_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Model saved")

# 2. Feature names
with open(f'{export_dir}/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("✓ Feature names saved")

# 3. Config — ablation + cross-dataset only (no stats test)
config = {
    'normalized_weights': normalized_weights,
    'max_scores':         max_scores_data,
    'thresholds':         thresholds,
    'anomaly_features':   anomaly_features,
    'ablation_results':   ablation_df.to_dict('records'),
    'cross_dataset':      cross_dataset_results,
}
with open(f'{export_dir}/config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("✓ Config saved (includes ablation + cross-dataset)")

# 4. Scaler
with open(f'{export_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved")

# 5. Encoders
encoders = {}
if 'card1' in df.columns:
    le_card1 = LabelEncoder()
    le_card1.fit(df['card1'].astype(str))
    encoders['card1_encoder'] = le_card1
    print("✓ card1 encoder saved")
if 'card2' in df.columns:
    encoders['card2_median'] = df['card2'].median()
    print("✓ card2 median saved")
with open(f'{export_dir}/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("✓ Encoders saved")

# 6. Metadata
metadata = {
    'model_type':    type(best_model).__name__,
    'accuracy':      best_acc,
    'auc':           best_auc,
    'features':      feature_cols,
    'created':       pd.Timestamp.now().isoformat(),
    'total_samples': len(df),
    'fraud_rate':    df['isFraud'].mean()
}
with open(f'{export_dir}/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✓ Metadata saved")

# 7. SHAP explainer
with open(f'{export_dir}/shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)
print("✓ SHAP explainer saved")

# ── Final summary ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ✅ ALL FILES READY — PAPER + FLASK API VERSION")
print("=" * 70)
print(f"\nExported to: {export_dir}")
for file in sorted(os.listdir(export_dir)):
    size = os.path.getsize(f'{export_dir}/{file}')
    print(f"  • {file:<30} ({size:,} bytes)")

shutil.make_archive('/content/cybercrime_models_v2', 'zip', export_dir)
files.download('/content/cybercrime_models_v2.zip')
print("\n✅ Download started!")
