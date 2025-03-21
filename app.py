from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("lead_scoring_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Receive JSON input
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"lead_score": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)