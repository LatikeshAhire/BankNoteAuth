import pickle
import pandas as pd
import numpy as np
import streamlit as st



model=pickle.load(open('model.pkl','rb'))


def root():
    return "Welcome All"


def predict_note(var,sk,ct,et):
  predict=model.predict([[var,sk,ct,et]])
  print(predict)
  return predict

  
  
def main():
  st.title("Bank Authenticator")
  html_temp="""
  <div style="background-color:tomato;padding:10px">
  <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
  </div>
  """
  Note=""
  st.markdown(html_temp,unsafe_allow_html=True)
  var=st.text_input("Variance","Type Here")
  sk=st.text_input("skewness","Type Here")
  ct=st.text_input("curtosis","Type Here")
  et=st.text_input("entropy","Type Here")
  result=""
  if st.button("Predict"):
    result=predict_note(var,sk,ct,et)
    if result[0]==0:
      Note="The note is Fake"
    else:
      Note="The note is Real"
  st.success(Note)
  if st.button("About"):
    st.text("Predicting Bank Note Authentication")
    st.text("Built with Streamlit")
    

if __name__=='__main__':
  main()
