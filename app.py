# app.py

from flask import Flask, request, render_template
import joblib
import numpy as np

model=joblib.load('random_regression')


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Map form inputs explicitly to features
    age = int(request.form.get('Age'))
    sex = int(request.form.get('Sex'))
    bmi = float(request.form.get('Bmi'))
    children = int(request.form.get('children'))
    smoker = int(request.form.get('smoker'))
    region = int(request.form.get('region'))

    # Create a feature list in the correct order
    feature_list = [age, sex, bmi, children, smoker, region]
    final_features = [np.array(feature_list)]  # Convert to NumPy array for the model

    # Make prediction
    prediction = model.predict(final_features)

    # Print the raw predicted value to the console
    formatted_prediction = f"Prediction Value: {prediction[0]:.2f}"
    # Render template with the raw prediction value
    return render_template('index.html', prediction_text=formatted_prediction)


if __name__ == "__main__":
    app.run(debug=True)
