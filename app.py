from flask import Flask, render_template, request
import joblib
import numpy as np
import os, sys
from exception import SensorException
from logger import logging

### loading model and scalar object
logging.info("Reading model object from model.sav file")
model=joblib.load('model_web_etc.sav')
logging.info("Reading transformer object from scaler.sav file")
scalar=joblib.load('scalar_web.sav')

app=Flask(__name__)

### creating home route
logging.info("Creating Home route")
@app.route("/")
def home():
    return render_template("home.html")


# Creating Prediction route
logging.info("Creating Prediction route")
@app.route('/prediction',methods=['POST'])
def prediction():
    try:
        logging.info("Getting data from the web form for prediction")
        data=[float(x) for x in request.form.values()]
        logging.info("Appling standard scaling to input data and reshaping data for single prediction")
        final_input=scalar.transform(np.array(data).reshape(1,-1))
        logging.info("Passing input to model and getting prediction")
        output=model.predict(final_input)[0]
        #print(output)
        if output==1:
            logging.info("Model Prediction is: Tumor is Malignant. Your diagnosis is Positive.")
            output= "Tumor is Malignant. Your diagnosis is Positive."
        else:
            logging.info("Model Prediction is: Tumor is Benign. Your diagnosis is Negative.")
            output="Tumor is Benign. Your diagnosis is Negative."
        logging.info("Returning model prediction to web application")
        return render_template("home.html",prediction_value="Model Prediction: {}".format(output))
    except Exception as e:
        raise SensorException(e, sys)

### Note check log file for the server link or paste this in browser: http://127.0.0.1:5000
if __name__=='__main__':
    app.run()