from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Step 4: Create Flask app
app = Flask(__name__)

# Step 5: Load the trained model
# (This happens once when the server starts, making predictions faster)
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Step 6: Define the exact feature list used during training
FEATURES = [
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Rate",
    "Srate",
    "ack_count",
    "syn_count",
    "rst_count",
    "Tot size",
    "IAT"
]

# Step 7: Create a simple "/" route for testing
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "IoT DDoS Detection API is running!",
        "status": "active"
    })

# Step 8: Create the "/predict" POST route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the user's request
        data = request.get_json()

        # Error Handling: Check if no data was sent
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Error Handling: Check for missing features
        missing_features = [feat for feat in FEATURES if feat not in data]
        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing": missing_features
            }), 400

        # Format data into a dictionary, ensuring the order matches FEATURES
        input_dict = {feat: [data[feat]] for feat in FEATURES}
        
        # Convert to Pandas DataFrame (required by scikit-learn models)
        input_df = pd.DataFrame(input_dict)

        # Make the prediction
        prediction_array = model.predict(input_df)
        prediction_value = int(prediction_array[0])

        # Map the numeric output to a human-readable string
        if prediction_value == 0:
            result_text = "Normal"
        else:
            result_text = "Attack"

        # Return the JSON response
        return jsonify({
            "prediction_code": prediction_value,
            "prediction_label": result_text,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask server
    app.run(debug=True, host="0.0.0.0", port=5000)