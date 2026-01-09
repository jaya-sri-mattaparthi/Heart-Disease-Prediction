import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

rf_model = pickle.load(open('models/rf.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    result_message = None
    if request.method == 'POST':
        try:
            Age = float(request.form.get('Age'))
            sex = float(request.form.get('sex'))
            cp = float(request.form.get('cp'))
            trestbps = float(request.form.get('trestbps'))
            chol = float(request.form.get('chol'))
            fbs = float(request.form.get('fbs'))
            restecg = float(request.form.get('restecg'))
            thalach = float(request.form.get('thalach'))
            exang = float(request.form.get('exang'))
            oldpeak = float(request.form.get('oldpeak'))
            slope = float(request.form.get('slope'))
            ca = float(request.form.get('ca'))
            thal = float(request.form.get('thal'))

            new_data_scaled = standard_scaler.transform([[Age, sex, cp, trestbps, chol, fbs, restecg,
                                                         thalach, exang, oldpeak, slope, ca, thal]])
            result = rf_model.predict(new_data_scaled)[0]
            if result == 1:
                result_message = {
                    "text": "Heart Disease detected",
                    "type": "danger"
                }
            else:
                result_message = {
                    "text": "No Heart Disease detected",
                    "type": "success"
                }
        except Exception as e:
            result_message = {
                "text": f"Error in input: {str(e)}",
                "type": "danger"
            }
    return render_template('index.html', result_message=result_message)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
