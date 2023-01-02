# Breast Cancer Prediction

### Problem Statement
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

This is a Binary Classification problem, in which the affirmative class indicates that the Person has Malignant tumor i.e Breast Cancer, while the negative class
indicates that the Person has Benign tumor i.e No Breast Cancer.

### Features
1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter^2 / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension
11. Ten additional features to show Standard Error in above features.
12. Ten additional features to show worst case values for each record in each features.
13. Diagnosis (M = malignant, B = benign)

### Solution Proposed 
In this project, Positive class corresponds to the Person has Malignant tumor i.e Breast Cancer and Negative class corresponds to the Person has Benign tumor i.e No Breast Cancer.

The aim is to correctly classify the patients with cancer and reduce False Negative.
False Negative => The person has cancer but model prediction is Negative(No Cancer).

## Tech Stack Used
1. Python 
2. FlaskAPI 
3. Machine learning algorithms
4. HTML and CSS

## Infrastructure Required.
1. VS Code (you can use any other IDE)
2. GIT
3. GITHUB

## How to run?
This app was deployed on Heroku, but as the heroku is closed all free tiers, This app is now available for offline run.
In comming days, i'll deploy this app on AWS.


### Step 1: Clone the repository
```bash
git clone https://github.com/Shubham3023/BreastCancerPrediction.git
```

### Step 2- Download the code to local and use any IDE for running the code.

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```


### Step 4.A - Run the application server for web app
```bash
1. Run app.py
```

```bash
2. Open this link in browser: http://127.0.0.1:5000
```

### Step 4.B. For batch prediction
```
1. Provide input file path for start_batch_prediction function
2. Run batch_prediction.py file 
3. Prediction file is saved in Batch Prediction directory


```


## Application
![image](https://github.com/Shubham3023/BreastCancerPrediction/blob/main/Notebook/Homeapp.jpg)

## Positive Prediction
![image](https://github.com/Shubham3023/BreastCancerPrediction/blob/main/Notebook/Positive%20Prediction.jpg)

## Negative Prediction
![image](https://github.com/Shubham3023/BreastCancerPrediction/blob/main/Notebook/Negative%20Prediction.jpg)
