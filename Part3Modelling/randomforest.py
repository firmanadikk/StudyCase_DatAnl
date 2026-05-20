"""
============================================================
PART 3 – CHURN PREDICTION MODELLING
Studi Kasus Data Analyst – Astra Integrasi Digital (AID)
Model: Random Forest Classifier
============================================================

Alur kerja:
  1. Load & merge datasets
  2. Feature Engineering
  3. Preprocessing & Train-Test Split
  4. Model Training (Random Forest)
  5. Evaluasi (Accuracy, Precision, Recall, AUC)
  6. Feature Importance & Interpretasi
  7. Segmentasi Risiko (High / Medium / Low)
============================================================
"""

# ── 0. IMPORT LIBRARY ────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)

users        = pd.read_csv("user.csv")
activity     = pd.read_csv("activity.csv")
transactions = pd.read_csv("transactions.csv")
status       = pd.read_csv("status_clean.csv")

activity["event_date"]           = pd.to_datetime(activity["event_date"])
users["signup_date"]             = pd.to_datetime(users["signup_date"])
transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

print(f"  Users        : {users.shape}")
print(f"  Activity     : {activity.shape}")
print(f"  Transactions : {transactions.shape}")
print(f"  Status       : {status.shape}")


# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

# ---- Fitur dari Activity ----
act_feat = activity.groupby("user_id").agg(
    total_sessions   = ("session_count", "sum"),    # total sesi login
    avg_sessions_day = ("session_count", "mean"),   # rata-rata sesi per hari aktif
    active_days      = ("event_date", "nunique"),   # jumlah hari unik aktif
    unique_features  = ("feature_used", "nunique"), # ragam fitur yang dipakai
).reset_index()

# One-hot: fitur aplikasi yang pernah digunakan
feat_pivot = (
    pd.get_dummies(activity[["user_id", "feature_used"]], columns=["feature_used"])
    .groupby("user_id").max().reset_index()
)

# ---- Fitur dari Transactions ----
txn_feat = transactions.groupby("user_id").agg(
    total_amount   = ("amount", "sum"),    # total nominal pinjaman
    avg_amount     = ("amount", "mean"),   # rata-rata pinjaman
    max_amount     = ("amount", "max"),    # pinjaman terbesar
    loan_count_txn = ("amount", "count"),  # jumlah transaksi
).reset_index()

# ---- Fitur dari Users ----
SNAPSHOT_DATE    = pd.Timestamp("2025-06-01")
users["tenure_days"] = (SNAPSHOT_DATE - users["signup_date"]).dt.days

chan_dummies = pd.get_dummies(
    users[["user_id", "acquisition_channel"]], columns=["acquisition_channel"]
)

# ---- Gabungkan semua fitur ----
df = status[["user_id", "loan_count", "avg_transaction_amount",
             "days_since_last_activity", "is_churn"]].copy()

df = (
    df
    .merge(act_feat,   on="user_id", how="left")
    .merge(txn_feat,   on="user_id", how="left")
    .merge(feat_pivot, on="user_id", how="left")
    .merge(users[["user_id", "age", "tenure_days"]], on="user_id", how="left")
    .merge(chan_dummies, on="user_id", how="left")
)

df = df.fillna(0)  # user tanpa data aktivitas/transaksi = 0

print(f"  Dataset final : {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"  Churn (1)     : {df['is_churn'].sum()} pengguna ({df['is_churn'].mean()*100:.1f}%)")
print(f"  Aktif (0)     : {(df['is_churn']==0).sum()} pengguna ({(1-df['is_churn'].mean())*100:.1f}%)")


# ── 3. TRAIN-TEST SPLIT ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: TRAIN-TEST SPLIT (80% / 20%)")
print("=" * 60)

FEATURE_COLS = [c for c in df.columns if c not in ("user_id", "is_churn")]
TARGET_COL   = "is_churn"

X = df[FEATURE_COLS].astype(float)
y = df[TARGET_COL]

# stratify=y → menjaga proporsi churn di train dan test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Fitur         : {len(FEATURE_COLS)}")
print(f"  Train set     : {X_train.shape[0]} data")
print(f"  Test set      : {X_test.shape[0]} data")


# ── 4. TRAINING – RANDOM FOREST ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: TRAINING – RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200,        # 200 decision tree
    max_depth=6,             # kedalaman max tiap pohon (cegah overfitting)
    min_samples_leaf=5,      # min sampel di setiap leaf (cegah overfitting)
    class_weight="balanced", # kompensasi imbalance: churn 30% vs aktif 70%
    random_state=42
)

rf.fit(X_train, y_train)
print("  [✓] Random Forest berhasil dilatih")
print(f"  Jumlah pohon  : {rf.n_estimators}")
print(f"  Max depth     : {rf.max_depth}")


# ── 5. EVALUASI MODEL ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: EVALUASI MODEL")
print("=" * 60)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)

# 5-fold cross-validation untuk validasi kestabilan model
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

print(f"\n  Accuracy   : {acc:.4f}   → seberapa sering prediksi benar secara keseluruhan")
print(f"  Precision  : {prec:.4f}   → dari yang diprediksi churn, berapa % benar-benar churn")
print(f"  Recall     : {rec:.4f}   → dari semua churn aktual, berapa % berhasil terdeteksi")
print(f"  F1-Score   : {f1:.4f}   → harmonic mean precision & recall")
print(f"  AUC-ROC    : {auc:.4f}   → kemampuan model memisahkan churn vs aktif")
print(f"  CV AUC     : {cv_auc:.4f}   → rata-rata AUC dari 5-fold cross validation")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Aktif", "Churn"]))


# ── 6. FEATURE IMPORTANCE ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: FEATURE IMPORTANCE")
print("=" * 60)

fi_df = (
    pd.DataFrame({"feature": FEATURE_COLS, "importance": rf.feature_importances_})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
fi_df["rank"] = fi_df.index + 1

print("\n  Top 10 Driver Churn:")
print(fi_df[["rank", "feature", "importance"]].head(10).to_string(index=False))


# ── 7. SEGMENTASI RISIKO ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: SEGMENTASI RISIKO PENGGUNA")
print("=" * 60)

X_all = df[FEATURE_COLS].astype(float)
df["churn_prob"] = rf.predict_proba(X_all)[:, 1]

# Threshold segmentasi:
# >= 0.60 → High-Risk  : intervensi segera
# 0.30–0.59 → Medium-Risk : monitoring aktif
# < 0.30  → Low-Risk   : pertahankan dengan loyalty program
df["risk_segment"] = pd.cut(
    df["churn_prob"],
    bins=[0, 0.30, 0.60, 1.0],
    labels=["Low-Risk", "Medium-Risk", "High-Risk"],
    include_lowest=True
)

seg_summary = df.groupby("risk_segment", observed=True).agg(
    jumlah_user          = ("user_id",                  "count"),
    pct_churn_aktual     = ("is_churn",                 "mean"),
    avg_loan_count       = ("loan_count",               "mean"),
    avg_days_inactive    = ("days_since_last_activity", "mean"),
    avg_transaction_amt  = ("avg_transaction_amount",   "mean"),
).round(2)

print("\n  Ringkasan Segmen Risiko:")
print(seg_summary.to_string())

seg_out = df[["user_id", "is_churn", "churn_prob", "risk_segment",
              "loan_count", "days_since_last_activity", "avg_transaction_amount"]]
seg_out.to_csv("part3_risk_segmentation.csv", index=False)
print("\n  [✓] Segmentasi disimpan: part3_risk_segmentation.csv")


# ── 8. VISUALISASI ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8: MEMBUAT VISUALISASI")
print("=" * 60)

COLORS = {"churn": "#E74C3C", "active": "#2ECC71", "rf": "#DD8452",
          "bar": "#4C72B0", "high": "#E74C3C", "med": "#F39C12", "low": "#2ECC71"}

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Part 3 – Churn Prediction Modelling (Random Forest)\nAstra Integrasi Digital (AID)",
    fontsize=15, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)


# ── Plot 1: Confusion Matrix ──
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
ax1.imshow(cm, cmap="Oranges")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Aktif", "Churn"])
ax1.set_yticks([0, 1]); ax1.set_yticklabels(["Aktif", "Churn"])
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix", fontweight="bold")
for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black",
                 fontsize=18, fontweight="bold")

# Label annotation per kuadran
labels = [["TN\n(Benar Aktif)", "FP\n(Salah Churn)"],
          ["FN\n(Terlewat)", "TP\n(Benar Churn)"]]
for i in range(2):
    for j in range(2):
        ax1.text(j, i + 0.35, labels[i][j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black",
                 fontsize=7)


# ── Plot 2: ROC Curve ──
ax2 = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax2.plot(fpr, tpr, color=COLORS["rf"], linewidth=2.5,
         label=f"Random Forest (AUC = {auc:.3f})")
ax2.fill_between(fpr, tpr, alpha=0.15, color=COLORS["rf"])
ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline (AUC = 0.5)")
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve", fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)


# ── Plot 3: Metrik Evaluasi ──
ax3 = fig.add_subplot(gs[0, 2])
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
metric_vals  = [acc, prec, rec, f1, auc]
bar_colors   = [COLORS["bar"] if v >= 0.6 else COLORS["churn"] for v in metric_vals]
bars = ax3.bar(metric_names, metric_vals, color=bar_colors, alpha=0.85, edgecolor="white")
for bar in bars:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
ax3.set_ylim(0, 1.15)
ax3.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Baseline 0.5")
ax3.set_ylabel("Score"); ax3.set_title("Metrik Evaluasi", fontweight="bold")
ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)
ax3.tick_params(axis="x", rotation=15)


# ── Plot 4: Feature Importance (top 12) ──
ax4 = fig.add_subplot(gs[1, :2])
top12 = fi_df.head(12).sort_values("importance", ascending=True)
thresh = top12["importance"].quantile(0.6)
bar_fi_colors = [COLORS["churn"] if v >= thresh else COLORS["bar"]
                 for v in top12["importance"]]
bars4 = ax4.barh(top12["feature"], top12["importance"],
                 color=bar_fi_colors, alpha=0.85, edgecolor="white")
for bar in bars4:
    ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.4f}", va="center", fontsize=8)
ax4.set_xlabel("Importance Score")
ax4.set_title("Top 12 Feature Importance\n(Merah = driver churn terkuat)", fontweight="bold")
ax4.grid(axis="x", alpha=0.3)


# ── Plot 5: Distribusi Probabilitas Churn ──
ax5 = fig.add_subplot(gs[1, 2])
prob_churn  = y_prob[y_test.values == 1]
prob_active = y_prob[y_test.values == 0]
ax5.hist(prob_active, bins=14, alpha=0.65, color=COLORS["active"],
         label=f"Aktif (n={len(prob_active)})", edgecolor="white")
ax5.hist(prob_churn, bins=14, alpha=0.65, color=COLORS["churn"],
         label=f"Churn  (n={len(prob_churn)})", edgecolor="white")
ax5.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold 0.5")
ax5.axvspan(0.60, 1.0, alpha=0.08, color=COLORS["churn"], label="High-Risk Zone (≥0.6)")
ax5.set_xlabel("Probabilitas Churn yang Diprediksi")
ax5.set_ylabel("Jumlah Pengguna")
ax5.set_title("Distribusi Probabilitas Prediksi", fontweight="bold")
ax5.legend(fontsize=8); ax5.grid(alpha=0.3)


plt.savefig("part3_rf_results.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("  [✓] Visualisasi disimpan: part3_rf_results.png")


# ── 9. RINGKASAN AKHIR ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RINGKASAN AKHIR – RANDOM FOREST CHURN MODEL")
print("=" * 60)
print(f"""
  PERFORMA MODEL:
  ┌──────────────┬────────┐
  │ Accuracy     │ {acc:.4f} │
  │ Precision    │ {prec:.4f} │
  │ Recall       │ {rec:.4f} │
  │ F1-Score     │ {f1:.4f} │
  │ AUC-ROC      │ {auc:.4f} │
  │ CV AUC (5K)  │ {cv_auc:.4f} │
  └──────────────┴────────┘

  TOP 5 DRIVER CHURN:
{chr(10).join(f"  {r.rank}. {r.feature:<30} importance: {r.importance:.4f}" for r in fi_df.head(5).itertuples())}

  SEGMENTASI RISIKO:
{seg_summary[['jumlah_user','pct_churn_aktual','avg_days_inactive']].to_string()}

  FILE OUTPUT:
  - part3_rf_results.png          → visualisasi model
  - part3_risk_segmentation.csv   → prediksi risiko per user
""")