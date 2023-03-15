from flask import Flask,request,jsonify
import numpy as np
import pickle
import joblib
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    cp = request.form.get('cp')
    trtbps = request.form.get('trtbps')
    fbs = request.form.get('fbs')

    input_query = np.array([[cp,trtbps,fbs]])

    result = model.predict(input_query)[0]

    return jsonify({'placement':str(result)})

if __name__ == '__main__':
    app.run(debug=True)