
from transformers import pipeline, set_seed
from flask import Flask, request, jsonify

set_seed(42)
generator = pipeline("text-generation", model="distilgpt2")

app = Flask(__name__)

@app.route("/generate_insight", methods=["POST"])
def generate_insight():
    try:
        data = request.json
        credit_score = data["credit_score"]
        risk = data["risk"]
        recommendation = data.get("recommendation", "")

        prompt = f"Customer Credit Score: {credit_score}\nRisk Profile: {risk}\n{recommendation}\nFinancial Insight:"
        result = generator(prompt, max_length=100, num_return_sequences=1)
        return jsonify({
            "prompt": prompt,
            "generated_insight": result[0]["generated_text"]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5001)
