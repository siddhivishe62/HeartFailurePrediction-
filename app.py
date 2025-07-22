from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model (no scaler if you didn't use it)
model = pickle.load(open('model.pkl', 'rb'))
scaler = None  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array([data])
    output = model.predict(final_input)
    prediction = 'Risk of Death' if output[0] == 1 else 'No Risk of Death'
    return render_template('index.html', prediction_text=f'Prediction: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
