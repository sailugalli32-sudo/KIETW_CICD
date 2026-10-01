# app.py
from flask import Flask, request, jsonify, render_template
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model from the saved file
model = joblib.load("iris_model.pkl")

# Initialize the Flask application
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Render the HTML file from templates/

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract JSON data from the incoming POST request
        data = request.get_json(force=True)
        
        # Validate the input: Check if 'features' key exists and contains exactly 4 numerical values
        if "features" not in data or len(data["features"]) != 4:
            return jsonify({"error": "Exactly 4 numerical features are required"}), 400
        
        # Convert the features list to a NumPy array and reshape it to (1, 4) for single-sample prediction
        features = np.array(data["features"], dtype=float).reshape(1, -1)
        
        # Use the model to predict the class (0, 1, or 2)
        prediction = model.predict(features)[0]
        
        # Map the numerical prediction to the corresponding species name
        classes = ["setosa", "versicolor", "virginica"]
        result = {"prediction": classes[prediction]}
        
        # Return the result as a JSON response with HTTP status 200 (OK)
        return jsonify(result)
    except Exception as e:
        # Handle any errors (e.g., invalid data types) and return an error message with HTTP status 400 (Bad Request)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Run the server on all interfaces (for external access) on port 5000