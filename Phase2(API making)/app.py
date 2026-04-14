from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model once at startup
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

# Exact 36-feature order used in training
FEATURES = [
    "IAT", "Min", "Magnitue", "fin_flag_number", "psh_flag_number", "syn_flag_number",
    "Tot sum", "Protocol Type", "ICMP", "Header_Length", "rst_count", "Radius",
    "fin_count", "syn_count", "flow_duration", "Srate", "Number", "AVG", "Rate",
    "Variance", "HTTPS", "urg_count", "Duration", "Weight", "HTTP", "Max", "Tot size",
    "Covariance", "ack_count", "Std", "rst_flag_number", "UDP", "ack_flag_number",
    "SSH", "TCP", "LLC"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "IoT DDoS Detection API is running!",
        "status": "active"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Ensure all required 36 features are present
        missing_features = [feat for feat in FEATURES if feat not in data]
        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing": missing_features
            }), 400

        # Build input in exact training order
        input_dict = {feat: [data[feat]] for feat in FEATURES}
        input_df = pd.DataFrame(input_dict)

        # Force numeric conversion (safe for model inference)
        input_df = input_df.apply(pd.to_numeric, errors="coerce")

        # Check invalid numeric values
        if input_df.isnull().any().any():
            bad_cols = input_df.columns[input_df.isnull().any()].tolist()
            return jsonify({
                "error": "Some features are non-numeric or invalid",
                "invalid_features": bad_cols
            }), 400

        prediction_array = model.predict(input_df)  # expected shape (1,)
        prediction_value = int(prediction_array[0])

        # Backend mapping requirement: 1=Attack, 0=Normal
        result_text = "Attack" if prediction_value == 1 else "Normal"

        return jsonify({
            "prediction_code": prediction_value,
            "prediction_label": result_text,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
