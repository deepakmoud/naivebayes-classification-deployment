# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:42:45 2021

@author: deepak
"""

import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle
import pandas as pd
dataset= pd.read_csv('DATASET education.csv')
X = dataset.iloc[:, 0:8].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

app = Flask(__name__)
model = pickle.load(open('Naive_Bayes_education_model.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    
    '''
    For rendering results on HTML GUI
    '''
    percentage10 = int(request.args.get('percentage10'))
    percentage12 = int(request.args.get('percentage12'))
    BTechpercentage = int(request.args.get('BTechpercentage'))
    marks7sem = int(request.args.get('marks7sem'))
    marks6sem = int(request.args.get('marks6sem'))
    marks5sem = int(request.args.get('marks5sem'))
    finalperformance = int(request.args.get('finalperformance'))
    medium = int(request.args.get('medium'))
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    prediction = model.predict(sc.transform([[percentage10,percentage12, BTechpercentage, marks7sem, marks6sem,marks5sem, finalperformance, medium ]]))
    print(prediction)
    if prediction==[1]:
        output='Placed'
    else:
        output='Not Placed'
        
    return render_template('index.html', prediction_text='Model has predicted student will be : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
