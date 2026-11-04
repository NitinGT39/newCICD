
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
with open('model.pkl', 'rb') as file:
    artifacts = pickle.load(file)

model = artifacts['model']
scaler = artifacts['scaler']

# Home route
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "new Deployment done",
        "status": "API is running successfully"
    })


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get input JSON
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Please provide JSON input"}), 400

        # Extract features
        features = data.get('features')

        if features is None:
            return jsonify({
                "error": "Missing features"
            }), 400

        if len(features) != 4:
            return jsonify({
                "error": "Exactly 4 features are required"
            }), 400

        # Convert input into numpy array
        input_data = np.array(features, dtype=float).reshape(1, -1)

        if not np.isfinite(input_data).all():
            return jsonify({
                "error": "Features must be finite numbers"
            }), 400

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)

        # Iris class names
        class_names = [
            "Setosa",
            "Versicolor",
            "Virginica"
        ]

        predicted_class = class_names[int(prediction[0])]

        return jsonify({
            "prediction": int(prediction[0]),
            "flower_name": predicted_class,
            "status": "success"
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Invalid feature values"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
