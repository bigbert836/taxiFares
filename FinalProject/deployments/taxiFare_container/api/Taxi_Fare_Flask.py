# -*- coding: utf-8 -*-
"""

@author: bab232
"""
import numpy as np
from flask import Flask, request, render_template
#TO generate UI for sending request via browser
from flasgger import Swagger 

import pickle
import pandas as pd

app = Flask(__name__)

#Enable this app for swagger and it will auto generate UI
swagger = Swagger(app)
model = pickle.load(open('../random-forest-model/model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Fare should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)  