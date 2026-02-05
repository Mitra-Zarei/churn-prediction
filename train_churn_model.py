import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from datetime import timedelta
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score


df = pd.read_csv("dataset.csv", sep=";", decimal=",")
df['calculation_date'] = pd.to_datetime(df['calculation_date'])

print(f" Rows: {len(df)}, Columns: {list(df.columns)}")

last_date = df["calculation_date"].max()
reference_date = last_date - timedelta(days=30)

# Only customers who have at least one row on reference_date or before reference_date
mask = df["calculation_date"] <= reference_date
befor_reference = df[mask]
last_dates = (
    befor_reference.groupby("customer_id")["calculation_date"]
    .max()
    .reset_index()
    .rename(columns={"calculation_date": "last_date"})
)

# Churn is "1" if no activity between reference_date and reference_date+30
future_mask = (
    (df["calculation_date"] > reference_date) &
    (df["calculation_date"] <= reference_date + timedelta(days=30))
)
activity_in_window = df.loc[future_mask, "customer_id"].unique()
last_dates["activity_in_next_30d"] = last_dates["customer_id"].isin(activity_in_window)

last_dates["churn"] = 0
last_dates.loc[~last_dates["activity_in_next_30d"], "churn"] = 1

print(f"  Reference date: {reference_date.date()}, Customers with label: {len(last_dates)}, Churn rate: {last_dates['churn'].mean():.2%}")

# Buid features based on customer history
def build_features(customer_id, last_date):
    customer_history = df[(df["customer_id"] == customer_id) & (df["calculation_date"] <= last_date)]
    if customer_history.empty:
        return None
    
    first_date = customer_history["calculation_date"].min()
    last_7d_activity = customer_history[customer_history["calculation_date"] > last_date - timedelta(days=7)]
    last_30d_activity = customer_history[customer_history["calculation_date"] > last_date - timedelta(days=30)]
    return {
        "customer_id": customer_id,
        "country": customer_history["country"].iloc[-1],
        "n_active_days": len(customer_history),
        "span_days": (last_date - first_date).days,
        "total_deposit": customer_history["deposit_EUR"].sum(),
        "total_gaming": customer_history["gaming_turnover_EUR"].sum(),
        "total_betting": customer_history["betting_turnover_EUR"].sum(),
        "total_withdrawal": customer_history["withdrawal_EUR"].sum(),
        "total_logins": customer_history["login_count"].sum(),
        "deposit_last_7d": last_7d_activity["deposit_EUR"].sum(),
        "deposit_last_30d": last_30d_activity["deposit_EUR"].sum(),
        "logins_last_7d": last_7d_activity["login_count"].sum(),
        "logins_last_30d": last_30d_activity["login_count"].sum(),
        "gaming_last_30d": last_30d_activity["gaming_turnover_EUR"].sum(),
    }

feature_rows = []
for _, row in last_dates.iterrows():
    new_feature = build_features(row["customer_id"], row["last_date"])
    if new_feature is not None:
        new_feature["churn"] = row["churn"]
        new_feature["last_date"] = row["last_date"]
        feature_rows.append(new_feature)

data = pd.DataFrame(feature_rows)
print(f"  Feature matrix: {data.shape}")

# earlier dates for training and later dates for testing
data = data.sort_values("last_date").reset_index(drop=True)
split_idx = int(len(data) * 0.8)
train_df = data.iloc[:split_idx]
test_df = data.iloc[split_idx:]

cat_col = "country"
num_cols = [
    "n_active_days", "span_days",
    "total_deposit", "total_gaming", "total_betting", "total_withdrawal", "total_logins",
    "deposit_last_7d", "deposit_last_30d", "logins_last_7d", "logins_last_30d", "gaming_last_30d",
]
feature_cols = [cat_col] + num_cols

X_train = train_df[feature_cols]
y_train = train_df["churn"]
X_test = test_df[feature_cols]
y_test = test_df["churn"]

for c in [cat_col]:
    X_train[c] = X_train[c].astype("category")
    X_test[c] = X_test[c].astype("category")

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "verbosity": -1,
    "seed": 42,
}
train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=[cat_col])
model = lgb.train(
    params,
    train_set,
    num_boost_round=200,
    valid_sets=[lgb.Dataset(X_test, label=y_test, reference=train_set)],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
)

# Evaluate and plot 

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)
auc_val = roc_auc_score(y_test, y_pred_proba) if y_test.nunique() > 1 else float("nan")
acc_val = accuracy_score(y_test, y_pred)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

ax1 = axes[0]
if y_test.nunique() > 1:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax1.plot(fpr, tpr, color="red", lw=2, label=f"ROC (AUC = {auc_val:.3f})")
ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
im = ax2.imshow(cm, cmap="Blues")
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(["No churn", "Churn"])
ax2.set_yticklabels(["No churn", "Churn"])
ax2.set_xlabel("Predicted")
ax2.set_ylabel("True")
ax2.set_title(f"Confusion Matrix (Accuracy = {acc_val:.3f})")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
plt.colorbar(im, ax=ax2, label="Count")

ax3 = axes[2]
imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=True)
ax3.barh(imp["feature"], imp["importance"], color="blue", alpha=0.8)
ax3.set_xlabel("Importance (gain)")
ax3.set_title("Feature Importance")
ax3.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig("churn_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plots saved to churn_evaluation.png")

model.save_model("churn_model.txt")
print("Done. Model saved to churn_model.txt")
