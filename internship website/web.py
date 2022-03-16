# -*- coding: utf-8 -*-

"""
Created on Thu Nov 11 11:47:54 2021

@author: SAMSUNG
"""

from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
   
   input_x = [float(x) for x in request.form.values()]
   
   x_values = [np.array(input_x)]
   output = clf.predict(x_values)
   output=output.item()
 
   if output == 0:
       output = "very low"
       rank=4
   elif output == 1:
       output = " mid"
       rank =3
   elif output == 2:
       output = " high"
       rank =2
   elif output == 3:
       output =" very high"  
       rank=1
   else:
       output="Not a Valid Entry"
       
   return render_template('result.html',prediction_text="the mobile price range is {} range and rank is {}".format(output,rank))


 
import webbrowser
from threading import Timer



def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
      Timer(1, open_browser).start();
      app.run(port=5000)    
      
      
 '''     
      
if __name__=='__main__':
    app.run(port=5000)





