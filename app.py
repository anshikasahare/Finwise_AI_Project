
from flask import Flask, request, jsonify
import joblib
import traceback

transaction_model = joblib.load("models/transaction_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
risk_model = joblib.load("models/credit_risk_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "FinWise API is running."

@app.route("/predict_category", methods=["POST"])
def predict_category():
    try:
        data = request.json
        text = data["description"]
        clean_text = ''.join(e for e in text.lower() if e.isalnum() or e.isspace())
        vector = vectorizer.transform([clean_text])
        prediction = transaction_model.predict(vector)[0]
        return jsonify({"category": prediction})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    try:
        data = request.json
        features = [
            data["credit_score"],
            data["existing_loans"],
            data["utilization_ratio"],
            data["missed_payments_12m"],
            data["total_outstanding_debt"],
            data["debt_to_income_ratio"]
        ]
        prediction = risk_model.predict([features])[0]
        risk = "High Risk" if prediction == 1 else "Low Risk"
        return jsonify({"risk": risk})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    app.run(debug=True)
