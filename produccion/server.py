import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    X_new = np.array([7.59, 7.47, 1.61, 1.53, 0.79, 0.63, 0.36, 0.31, 2.27])
    prediction = model.predict(X_new.reshape(1, -1))

    return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)