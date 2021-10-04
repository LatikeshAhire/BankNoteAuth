from flask import Flask,request
import pickle
from flask.templating import render_template
import pandas as pd
import numpy as np
from flasgger import Swagger


app=Flask(__name__)
Swagger(app)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def root():
    return "Welcome All"

@app.route('/predict',methods=['Get'])
def predict():
    
    """ Bank Note Authentication
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    

    responses:
        200:
            description: The output values

    """
    
    
    inp=[request.args.get('variance'),
         request.args.get('skewness'),
         request.args.get('curtosis'),
         request.args.get('entropy')]
    pred=model.predict([inp])
    
    if pred==1:
        note="Not Fake"
    else:
        note="Fake"
    return "The note is "+note        

@app.route('/predict_file',methods=['POST'])
def predict_file():
    
    """ Lets authenticate the bank notes
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    
    responses:
        200:
            description: The output values
        
    """
    
    
    df_test=pd.read_csv(request.files.get('file'))
    predict=model.predict(df_test)
    
    return str(list(predict))    #if the response is 200 then this will be the output

if __name__=='__main__':
    app.run(host = "0.0.0.0", port = 8000)
    
    
# try using postman for testing

# url/apidocs  (http://127.0.0.1:5000/apidocs/)