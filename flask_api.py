from flask import Flask,request
import pickle
from flask.templating import render_template
import pandas as pd
import numpy as np


app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def root():
    return render_template('index.html')

@app.route('/predict',methods=['get'])
def predict():
    inp=[request.args.get('variance'),
         request.args.get('skewness'),
         request.args.get('curtosis'),
         request.args.get('entropy')]
    pred=model.predict([inp])
    
    if pred==1:
        note="Not Fake"
    else:
        note="Fake"
    return render_template('index.html',predict_text=note)        

@app.route('/predict_file',methods=['POST'])
def predict_file():
    df_test=pd.read_csv(request.files.get('file'))
    predict=model.predict(df_test)
    
    return "The predicted values are "+ str(list(predict))


if __name__=='__main__':
    app.run(host = "0.0.0.0", port = 8000)
    
    
# try using postman for testing