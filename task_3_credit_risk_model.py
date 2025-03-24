
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

df = pd.read_csv("data/credit_reports.csv")

def risk_label(row):
    if row['credit_score'] < 600 or row['missed_payments_12m'] > 2 or row['debt_to_income_ratio'] > 0.5:
        return 1
    return 0

df["risk_flag"] = df.apply(risk_label, axis=1)

features = [
    "credit_score", "existing_loans", "utilization_ratio",
    "missed_payments_12m", "total_outstanding_debt", "debt_to_income_ratio"
]
X = df[features]
y = df["risk_flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/credit_risk_model.pkl")
