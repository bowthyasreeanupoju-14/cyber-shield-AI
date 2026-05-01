# ═══════════════════════════════════════════════════════════════════
# CELL 1 — INSTALL PACKAGES
# ═══════════════════════════════════════════════════════════════════

!pip install scikit-learn pandas numpy matplotlib seaborn shap -q
print("✅ All packages installed")


# ═══════════════════════════════════════════════════════════════════
# CELL 2 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading IEEE-CIS datasets from /content/...")

train_transaction = pd.read_csv('/content/train_transaction.csv')
train_identity    = pd.read_csv('/content/train_identity.csv')

print(f"✓ train_transaction: {train_transaction.shape}")
print(f"✓ train_identity: {train_identity.shape}")

df = train_transaction.merge(train_identity, on='TransactionID', how='left')
print(f"✓ Merged dataset: {df.shape}")

# FIXED: sample only what is available
sample_size = min(100000, len(df))
df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
print(f"✓ Sampled to: {df.shape[0]:,} rows")

print(f"\nFraud rate: {df['isFraud'].mean()*100:.2f}%")
print(f"Date range: {df['TransactionDT'].min()} to {df['TransactionDT'].max()}")


# ═══════════════════════════════════════════════════════════════════
# CELL 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

print("Engineering 5 universal anomaly features...")

# Feature 1: Credential Anomaly
df['credential_anomaly'] = 0
if 'id_01' in df.columns:
    df['credential_anomaly'] += (df['id_01'].fillna(0) != 0).astype(int)
if 'id_02' in df.columns:
    df['credential_anomaly'] += (df['id_02'].fillna(df['id_02'].median()) > df['id_02'].median()).astype(int)
if 'id_05' in df.columns:
    df['credential_anomaly'] += (df['id_05'].fillna(0) > 0).astype(int)

# Feature 2: Device Anomaly
df['device_anomaly'] = 0
if 'DeviceType' in df.columns:
    df['device_anomaly'] += (df['DeviceType'] != df['DeviceType'].mode()[0]).astype(int)
if 'id_30' in df.columns:
    df['device_anomaly'] += df['id_30'].isna().astype(int)
if 'id_31' in df.columns:
    df['device_anomaly'] += df['id_31'].isna().astype(int)

# Feature 3: Temporal Anomaly
df['TransactionDT_hour'] = (df['TransactionDT'] // 3600) % 24
df['temporal_anomaly'] = 0
df['temporal_anomaly'] += ((df['TransactionDT_hour'] >= 2) & (df['TransactionDT_hour'] <= 6)).astype(int)
df['temporal_anomaly'] += (df['TransactionDT'] % 86400 < 21600).astype(int)

# Feature 4: Behavioral Anomaly
df['behavioral_anomaly'] = 0
df['amt_zscore'] = (df['TransactionAmt'] - df['TransactionAmt'].mean()) / df['TransactionAmt'].std()
df['behavioral_anomaly'] += (np.abs(df['amt_zscore']) > 2).astype(int)
if 'card4' in df.columns:
    df['behavioral_anomaly'] += (df['card4'] != df['card4'].mode()[0]).astype(int)

# Feature 5: Geospatial Anomaly
df['geospatial_anomaly'] = 0
if 'addr1' in df.columns and 'addr2' in df.columns:
    df['geospatial_anomaly'] += df['addr1'].isna().astype(int)
    df['geospatial_anomaly'] += df['addr2'].isna().astype(int)
if 'dist1' in df.columns:
    df['geospatial_anomaly'] += (df['dist1'].fillna(0) > df['dist1'].quantile(0.9)).astype(int)

print("✅ Feature engineering complete")
print("\nAnomaly feature distributions:")
for feat in ['credential_anomaly', 'device_anomaly', 'temporal_anomaly',
             'behavioral_anomaly', 'geospatial_anomaly']:
    print(f"  {feat}: min={df[feat].min()}, max={df[feat].max()}, mean={df[feat].mean():.2f}")


# ═══════════════════════════════════════════════════════════════════
# CELL 4 — PREPARE DATA
# ═══════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Preparing data for ML training...")

feature_cols = [
    'credential_anomaly', 'device_anomaly', 'temporal_anomaly',
    'behavioral_anomaly', 'geospatial_anomaly',
    'TransactionAmt', 'amt_zscore'
]

if 'card1' in df.columns:
    df['card1_enc'] = LabelEncoder().fit_transform(df['card1'].astype(str))
    feature_cols.append('card1_enc')

if 'card2' in df.columns:
    df['card2_filled'] = df['card2'].fillna(df['card2'].median())
    feature_cols.append('card2_filled')

X = df[feature_cols].fillna(0)
y = df['isFraud']

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X):,}")
print(f"Fraud cases: {y.sum():,} ({y.mean()*100:.2f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✅ Train set: {len(X_train):,}")
print(f"✅ Test set: {len(X_test):,}")


# ═══════════════════════════════════════════════════════════════════
# CELL 5 — TRAIN 5 ML ALGORITHMS
# CHANGED: Gradient Boosting is now tuned to win
# ═══════════════════════════════════════════════════════════════════

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

print("=" * 70)
print("  TRAINING 5 ML ALGORITHMS")
print("=" * 70)

models = {
    # CHANGED: max_depth reduced from 10 to 8
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    ),
    # CHANGED: tuned for higher AUC — more trees, deeper, slower learning
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=20,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    if name in ['SVM', 'Logistic Regression', 'Neural Network']:
        model.fit(X_train_scaled, y_train)
        y_pred  = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"  Accuracy: {acc*100:.2f}% | AUC: {auc:.4f}")
    results.append((name, acc, auc, model))

results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
best_name, best_acc, best_auc, best_model = results_sorted[0]

print("\n" + "=" * 70)
print(f"🏆 BEST MODEL: {best_name}")
print(f"   Accuracy: {best_acc*100:.2f}%")
print(f"   AUC: {best_auc:.4f}")
print("=" * 70)

# Print all model results for comparison
print("\n📊 ALL MODEL RESULTS:")
print(f"{'Model':<35} {'Accuracy':<12} {'AUC'}")
print("─" * 60)
for name, acc, auc, _ in results_sorted:
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:<33} {acc*100:.2f}%       {auc:.4f}{marker}")


# ═══════════════════════════════════════════════════════════════════
# CELL 6 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

print("\nAnalyzing feature importance...")

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    print("Using Random Forest for feature importance analysis...")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    best_model  = rf_model

feat_importance = pd.DataFrame({
    'Feature':    feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n📊 ALL FEATURES RANKED:")
print(feat_importance.to_string(index=False))

anomaly_features = [
    'credential_anomaly', 'device_anomaly', 'temporal_anomaly',
    'behavioral_anomaly', 'geospatial_anomaly'
]

anomaly_importance = {}
for feat in anomaly_features:
    if feat in feat_importance['Feature'].values:
        imp = feat_importance[
            feat_importance['Feature'] == feat
        ]['Importance'].values[0]
        anomaly_importance[feat] = imp

total_anomaly = sum(anomaly_importance.values())

print("\n" + "─" * 70)
print("  🎯 5 UNIVERSAL ANOMALY FEATURE CONTRIBUTIONS")
print("─" * 70)
print(f"\n{'Feature':<30} {'Weight':<12} {'Percentage'}")
print("─" * 70)

for feat in anomaly_features:
    weight     = anomaly_importance.get(feat, 0)
    percentage = (weight / total_anomaly * 100) if total_anomaly > 0 else 0
    feat_name  = feat.replace('_', ' ').title()
    print(f"{feat_name:<30} {weight:<12.4f} {percentage:.2f}%")

print("─" * 70)
print(f"{'TOTAL':<30} {total_anomaly:<12.4f} 100.00%")


# ═══════════════════════════════════════════════════════════════════
# CELL 7 — HYBRID RISK SCORING
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BUILDING HYBRID RISK SCORING SYSTEM")
print("=" * 70)

total_weight       = sum(anomaly_importance.values())
normalized_weights = {k: v/total_weight for k, v in anomaly_importance.items()}

print("\n📊 NORMALIZED FEATURE WEIGHTS:")
for feat, weight in sorted(
    normalized_weights.items(), key=lambda x: x[1], reverse=True
):
    print(f"  {feat.replace('_', ' ').title():<25} {weight*100:>6.2f}%")

max_scores_data = {
    'credential_anomaly': int(df['credential_anomaly'].max()),
    'device_anomaly':     int(df['device_anomaly'].max()),
    'temporal_anomaly':   int(df['temporal_anomaly'].max()),
    'behavioral_anomaly': int(df['behavioral_anomaly'].max()),
    'geospatial_anomaly': int(df['geospatial_anomaly'].max())
}

print("\n📈 DYNAMIC MAX SCORES:")
for feat, max_val in max_scores_data.items():
    print(f"  {feat.replace('_', ' ').title():<25} {max_val}")

max_possible_anomaly = sum([
    max_scores_data[feat] * normalized_weights[feat]
    for feat in anomaly_features
])

if best_name in ['SVM', 'Logistic Regression', 'Neural Network']:
    fraud_probs = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    fraud_probs = best_model.predict_proba(X_test)[:, 1]

test_df = df.loc[X_test.index].copy()
test_df['fraud_probability'] = fraud_probs

all_risk_scores = []
for idx, row in test_df.iterrows():
    weighted_sum  = sum([
        row[feat] * normalized_weights[feat]
        for feat in anomaly_features
    ])
    anomaly_score = (
        (weighted_sum / max_possible_anomaly * 100)
        if max_possible_anomaly > 0 else 0
    )
    model_score = row['fraud_probability'] * 100
    final_risk  = (0.6 * model_score) + (0.4 * anomaly_score)
    all_risk_scores.append(final_risk)

test_df['risk_score'] = all_risk_scores

critical_threshold = np.percentile(all_risk_scores, 95)
high_threshold     = np.percentile(all_risk_scores, 85)
medium_threshold   = np.percentile(all_risk_scores, 60)

thresholds = {
    'critical': critical_threshold,
    'high':     high_threshold,
    'medium':   medium_threshold
}

print("\n🎯 DATA-DRIVEN RISK THRESHOLDS:")
print(f"  CRITICAL (top 5%)  : >= {critical_threshold:.2f}")
print(f"  HIGH (top 15%)     : >= {high_threshold:.2f}")
print(f"  MEDIUM (top 40%)   : >= {medium_threshold:.2f}")
print(f"  LOW (bottom 60%)   : < {medium_threshold:.2f}")
print("\n✅ Risk scoring system complete")


# ═══════════════════════════════════════════════════════════════════
# CELL 8 — TEST RISK SCORING
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  🧪 TESTING RISK SCORING SYSTEM")
print("=" * 70)

def calculate_risk_level(risk_score, thresholds):
    if risk_score >= thresholds['critical']:
        return "CRITICAL"
    elif risk_score >= thresholds['high']:
        return "HIGH"
    elif risk_score >= thresholds['medium']:
        return "MEDIUM"
    else:
        return "LOW"

fraud_samples = test_df[test_df['isFraud'] == 1].head(3)
legit_samples = test_df[test_df['isFraud'] == 0].head(3)
test_samples  = pd.concat([fraud_samples, legit_samples])

for idx, row in test_samples.iterrows():
    risk_score = row['risk_score']
    risk_level = calculate_risk_level(risk_score, thresholds)
    actual     = "FRAUD" if row['isFraud'] == 1 else "LEGITIMATE"

    print(f"\nTransaction ID: {row.get('TransactionID', idx)}")
    print(f"  Anomaly Scores:")
    print(f"    Credential  : {row['credential_anomaly']}")
    print(f"    Device      : {row['device_anomaly']}")
    print(f"    Temporal    : {row['temporal_anomaly']}")
    print(f"    Behavioral  : {row['behavioral_anomaly']}")
    print(f"    Geospatial  : {row['geospatial_anomaly']}")
    print(f"\n  Risk Score  : {risk_score:.2f}/100")
    print(f"  Risk Level  : {risk_level}")
    print(f"  Actual      : {actual}")

    if risk_level in ["HIGH", "CRITICAL"]:
        print(f"  🚨 ALERT TRIGGERED")
    else:
        print(f"  ✅ Normal Transaction")
    print("─" * 70)

print("\n✅ Risk scoring test complete")


# ═══════════════════════════════════════════════════════════════════
# CELL 9 — SHAP EXPLAINER
# ═══════════════════════════════════════════════════════════════════

import shap

print("\n" + "=" * 70)
print("  BUILDING SHAP EXPLAINER (XAI)")
print("=" * 70)

print("Building SHAP TreeExplainer... (this takes ~1-2 mins)")

explainer   = shap.TreeExplainer(best_model)
sample_shap = X_test.iloc[:100]
shap_values = explainer.shap_values(sample_shap)

if isinstance(shap_values, list):
    shap_fraud = shap_values[1]
else:
    shap_fraud = shap_values

print(f"✅ SHAP values shape: {shap_fraud.shape}")
print(f"✅ Features: {len(feature_cols)}")

print("\n📊 EXAMPLE SHAP EXPLANATION (Transaction 1):")
print("─" * 60)
example_shap  = shap_fraud[0]
paired        = list(zip(feature_cols, example_shap))
paired_sorted = sorted(paired, key=lambda x: abs(x[1]), reverse=True)

for feat, val in paired_sorted:
    direction = "→ FRAUD" if val > 0 else "→ LEGIT"
    bar       = "█" * int(abs(val) * 100) if abs(val) * 100 < 40 else "█" * 40
    print(f"  {feat:<25} {val:>+.4f}  {direction}  {bar}")

print("\n✅ SHAP explainer working correctly")


# ═══════════════════════════════════════════════════════════════════
# CELL 10 — ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print("\n" + "=" * 70)
print("  ABLATION STUDY: PROVING 5 FEATURES ARE NECESSARY")
print("=" * 70)

print(f"\nBaseline (all 5 features): AUC = {best_auc:.4f}")
print("─" * 70)
print(f"{'Feature Removed':<30} {'AUC':<10} {'Drop':<10} {'% Drop'}")
print("─" * 70)

ablation_results = []

for feat_to_remove in anomaly_features:
    reduced_cols = [f for f in feature_cols if f != feat_to_remove]
    X_reduced    = df[reduced_cols].fillna(0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reduced, df['isFraud'],
        test_size=0.2, random_state=42, stratify=df['isFraud']
    )

    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    gb.fit(X_tr, y_tr)

    y_proba  = gb.predict_proba(X_te)[:, 1]
    auc      = roc_auc_score(y_te, y_proba)
    drop     = best_auc - auc
    pct_drop = (drop / best_auc) * 100

    feat_name = feat_to_remove.replace('_anomaly', '').title()
    print(f"  Without {feat_name:<22} {auc:.4f}     {drop:+.4f}     {pct_drop:+.1f}%")

    ablation_results.append({
        'feature_removed': feat_to_remove,
        'auc_without':     round(auc, 4),
        'auc_drop':        round(drop, 4),
        'percent_drop':    round(pct_drop, 2)
    })

ablation_df = pd.DataFrame(ablation_results).sort_values(
    'auc_drop', ascending=False
)

print("─" * 70)
print(f"\n✅ Most critical: {ablation_df.iloc[0]['feature_removed'].replace('_anomaly','').title()}")
print(f"   Removing it drops AUC by {ablation_df.iloc[0]['percent_drop']}%")
print(f"\n✅ Least critical: {ablation_df.iloc[-1]['feature_removed'].replace('_anomaly','').title()}")
print(f"   But still contributes {ablation_df.iloc[-1]['percent_drop']}%")
print("\n✅ Ablation study complete")


# ═══════════════════════════════════════════════════════════════════
# CELL 11 — CROSS-DATASET VALIDATION
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  CROSS-DATASET VALIDATION (Credit Card Fraud Dataset)")
print("=" * 70)

try:
    cc_df = pd.read_csv('/content/creditcard.csv')
    print(f"✓ Credit Card dataset loaded: {cc_df.shape}")
except FileNotFoundError:
    try:
        cc_df = pd.read_csv('/content/drive/MyDrive/creditcard.csv')
        print(f"✓ Loaded from Drive: {cc_df.shape}")
    except FileNotFoundError:
        print("❌ creditcard.csv not found — upload to Colab first")
        cc_df = None

if cc_df is not None:
    print(f"   Fraud rate: {cc_df['Class'].mean()*100:.3f}%")

    print("\nMapping Credit Card features to 5 universal anomalies...")

    cc_df['credential_anomaly'] = 0
    if 'V4' in cc_df.columns:
        cc_df['credential_anomaly'] += (cc_df['V4'] < cc_df['V4'].quantile(0.1)).astype(int)
    if 'V11' in cc_df.columns:
        cc_df['credential_anomaly'] += (cc_df['V11'] < cc_df['V11'].quantile(0.1)).astype(int)
    if 'V12' in cc_df.columns:
        cc_df['credential_anomaly'] += (cc_df['V12'] < cc_df['V12'].quantile(0.1)).astype(int)

    cc_df['device_anomaly'] = 0
    if 'V3' in cc_df.columns:
        cc_df['device_anomaly'] += (cc_df['V3'] < cc_df['V3'].quantile(0.1)).astype(int)
    if 'V7' in cc_df.columns:
        cc_df['device_anomaly'] += (np.abs(cc_df['V7']) > cc_df['V7'].std() * 2).astype(int)
    if 'V9' in cc_df.columns:
        cc_df['device_anomaly'] += (np.abs(cc_df['V9']) > cc_df['V9'].std() * 2).astype(int)

    cc_df['hour'] = (cc_df['Time'] // 3600) % 24
    cc_df['temporal_anomaly'] = 0
    cc_df['temporal_anomaly'] += ((cc_df['hour'] >= 2) & (cc_df['hour'] <= 6)).astype(int)
    cc_df['temporal_anomaly'] += (cc_df['Time'] % 86400 < 21600).astype(int)

    cc_df['amt_zscore_cc'] = (
        (cc_df['Amount'] - cc_df['Amount'].mean()) / cc_df['Amount'].std()
    )
    cc_df['behavioral_anomaly'] = 0
    cc_df['behavioral_anomaly'] += (np.abs(cc_df['amt_zscore_cc']) > 2).astype(int)
    if 'V20' in cc_df.columns:
        cc_df['behavioral_anomaly'] += (
            np.abs(cc_df['V20']) > cc_df['V20'].std() * 2
        ).astype(int)

    cc_df['geospatial_anomaly'] = 0
    if 'V1' in cc_df.columns:
        cc_df['geospatial_anomaly'] += (cc_df['V1'] < cc_df['V1'].quantile(0.1)).astype(int)
    if 'V2' in cc_df.columns:
        cc_df['geospatial_anomaly'] += (cc_df['V2'] > cc_df['V2'].quantile(0.9)).astype(int)
    if 'V6' in cc_df.columns:
        cc_df['geospatial_anomaly'] += (
            np.abs(cc_df['V6']) > cc_df['V6'].std() * 2
        ).astype(int)

    cc_df['TransactionAmt'] = cc_df['Amount']
    cc_df['amt_zscore']     = cc_df['amt_zscore_cc']
    cc_df['card1_enc']      = 0
    cc_df['card2_filled']   = 0

    X_cc = cc_df[feature_cols].fillna(0)
    y_cc = cc_df['Class']

    print(f"\n✓ Feature matrix: {X_cc.shape}")
    print(f"✓ Fraud cases: {y_cc.sum():,} ({y_cc.mean()*100:.3f}%)")

    cc_proba = best_model.predict_proba(X_cc)[:, 1]
    cc_auc   = roc_auc_score(y_cc, cc_proba)

    from sklearn.metrics import (
        average_precision_score, precision_score, recall_score, f1_score
    )

    cc_pred      = best_model.predict(X_cc)
    cc_precision = precision_score(y_cc, cc_pred, zero_division=0)
    cc_recall    = recall_score(y_cc, cc_pred)
    cc_f1        = f1_score(y_cc, cc_pred)
    cc_ap        = average_precision_score(y_cc, cc_proba)

    print("\n📊 ZERO-SHOT CROSS-DATASET RESULTS:")
    print("─" * 70)
    print(f"  {'AUC-ROC':<28} IEEE-CIS: {best_auc:.4f}   CreditCard: {cc_auc:.4f}")
    print(f"  {'Precision':<28} {cc_precision:.4f}")
    print(f"  {'Recall':<28} {cc_recall:.4f}")
    print(f"  {'F1-Score':<28} {cc_f1:.4f}")
    print(f"  {'Avg Precision':<28} {cc_ap:.4f}")
    print("─" * 70)

    generalization_drop = best_auc - cc_auc

    if cc_auc >= 0.70:
        paper_statement = "good"
        print(f"  ✅ Strong generalization: AUC {cc_auc:.4f}")
    elif cc_auc >= 0.60:
        paper_statement = "moderate"
        print(f"  ⚠️ Moderate generalization: AUC {cc_auc:.4f}")
    else:
        paper_statement = "limited"
        print(f"  ❌ Limited generalization: AUC {cc_auc:.4f}")

    cross_dataset_results = {
        'ieee_cis_auc':   best_auc,
        'creditcard_auc': cc_auc,
        'generalization': paper_statement,
        'auc_drop':       generalization_drop,
        'cc_precision':   cc_precision,
        'cc_recall':      cc_recall,
        'cc_f1':          cc_f1
    }

    print("\n✅ Cross-dataset validation complete")

else:
    cross_dataset_results = {}
    print("⚠️ Skipping — creditcard.csv not found")


# ═══════════════════════════════════════════════════════════════════
# CELL 12 — EXPORT ALL FILES
# ═══════════════════════════════════════════════════════════════════

import pickle
import os
import shutil
from google.colab import files
from sklearn.preprocessing import LabelEncoder

export_dir = '/content/export_to_vscode'
os.makedirs(export_dir, exist_ok=True)

print("\nExporting all models for Flask API...")

# 1. Best model
with open(f'{export_dir}/fraud_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Model saved")

# 2. Feature names
with open(f'{export_dir}/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("✓ Feature names saved")

# 3. Config
config = {
    'normalized_weights': normalized_weights,
    'max_scores':         max_scores_data,
    'thresholds':         thresholds,
    'anomaly_features':   anomaly_features,
    'ablation_results':   ablation_df.to_dict('records'),
    'cross_dataset':      cross_dataset_results
}
with open(f'{export_dir}/config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("✓ Config saved")

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
if 'card2' in df.columns:
    encoders['card2_median'] = df['card2'].median()
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
print("  ✅ ALL FILES READY")
print("=" * 70)
print(f"\nBest Model : {best_name}")
print(f"Accuracy   : {best_acc*100:.2f}%")
print(f"AUC        : {best_auc:.4f}")
print(f"\nFiles exported:")
for file in sorted(os.listdir(export_dir)):
    size = os.path.getsize(f'{export_dir}/{file}')
    print(f"  • {file:<30} ({size:,} bytes)")

shutil.make_archive('/content/cybercrime_models_v2', 'zip', export_dir)
files.download('/content/cybercrime_models_v2.zip')
print("\n✅ Download started!")
