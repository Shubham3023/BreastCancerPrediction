from exception import SensorException
from logger import logging
import pandas as pd
from datetime import datetime
import os, sys
import joblib

### loading model and scalar object
logging.info("Reading model object from model_batch.sav file")
model=joblib.load('model_batch.sav')
logging.info("Reading transformer object from scaler_batch.sav file")
scalar=joblib.load('scalar_batch.sav')


PREDICTION_DIR=os.path.join(os.getcwd(),"Batch_Prediction")


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path) 
        logging.info("Dropping Target column from input file if it exists")
        if "Diagnosis" in df.columns:
            test_df= df.drop("Diagnosis", axis =1)
        else: 
            test_df=df
        test_df=scalar.transform(test_df)
        prediction = model.predict(test_df)      
        df["Prediction"]=prediction
        df["Prediction_Cat"]=df["Prediction"].astype(str).replace("1", "M").replace("0", "B")



        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        print("Batch prediction successful")
    except Exception as e:
        raise SensorException(e, sys)


if __name__=="__main__":
    try:
        start_batch_prediction("Prediction.csv")
    except Exception as e:
        raise SensorException(e, sys)