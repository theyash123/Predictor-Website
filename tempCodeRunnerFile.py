from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        # Extract values
        features = [
        form['brand'],
        form['processor_brand'],
        form['processor_name'],
        form['processor_gnrtn'],
        int(form['ram_gb']),
        form['ram_type'],
        int(form['ssd']),
        int(form['hdd']),
        form['os'],
        form['os_bit'],
        int(form['graphic_card_gb']),
        float(form['weight']),
        form['warranty'],
        1 if form['Touchscreen'] == 'Yes' else 0,
        1 if form['msoffice'] == 'Yes' else 0
]


        # Label encode categorical values
        for i, col in enumerate(label_encoders):
            if col in label_encoders and isinstance(features[i], str):
                encoder = label_encoders[col]
                if features[i] in encoder.classes_:
                    features[i] = encoder.transform([features[i]])[0]
                else:
                    # handle unknown values by using most frequent label
                    features[i] = 0

        prediction = model.predict([features])[0]
        return render_template('index.html', prediction_text=f"Predicted Laptop Price: ₹{int(prediction):,}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)