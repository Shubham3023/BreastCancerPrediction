from flask import Flask, render_template, request
import joblib
import numpy as np

### loading model and scalar object
model=joblib.load('model_web_etc.sav')
scalar=joblib.load('scalar_web.sav')

app=Flask(__name__)

### creating home route
@app.route("/")
def home():
    return render_template("home.html")


# Creating Prediction route
@app.route('/prediction',methods=['POST'])
def prediction():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    output=model.predict(final_input)[0]
    #print(output)
    if output==1:
        output= "Tumor is Malignant. Your diagnosis is Positive."
    else:
        output="Tumor is Benign. Your diagnosis is Negative."
    return render_template("home.html",prediction_value="Model Prediction: {}".format(output))

if __name__=='__main__':
    app.run()