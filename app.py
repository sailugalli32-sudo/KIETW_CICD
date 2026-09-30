from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Load trained model
model = joblib.load("iris_model.pkl")

# Initialize Flask
app = Flask(__name__)


# Home route
@app.route("/", methods=["GET"])
def home():
    return """
    <h1>Iris Classifier API is Running!</h1>
    <p>Use POST /predict to make a prediction.</p>
    <p>Required JSON format:</p>
    <pre>
{
    "features": [5.1, 3.5, 1.4, 0.2]
}
    </pre>
    """


# Prediction route
@app.route("/predict", methods=["GET", "POST"])
def predict():

    # If opened directly in browser
    if request.method == "GET":
        return """
        <h2>Iris Prediction API</h2>
        <p>This endpoint requires a POST request.</p>
        <p>Send 4 flower measurements using JSON.</p>
        """

    try:
        data = request.get_json(force=True)

        if "features" not in data:
            return jsonify({
                "error": "Please provide 'features'"
            }), 400

        if len(data["features"]) != 4:
            return jsonify({
                "error": "Exactly 4 numerical features are required"
            }), 400

        features = np.array(
            data["features"],
            dtype=float
        ).reshape(1, -1)

        prediction = model.predict(features)[0]

        classes = [
            "setosa",
            "versicolor",
            "virginica"
        ]

        result = {
            "prediction": classes[int(prediction)]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


# Render server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port
    )