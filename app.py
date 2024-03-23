import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def predict():

    """ Selected feature are customerID, Dependents, tenure,gender, SeniorCitizen, 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'MonthlyCharges', MonthlyCharges, TotalCharges """
    customerID = float(request.form['customerID'])
    gender = request.form['gender']
    SeniorCitizen  = int(request.form['SeniorCitizen'])
    Partner = float(request.form['Partner'])
    Dependents = float(request.form['Dependents'])
    tenure = float(request.form['tenure'])
    PhoneService = float(request.form['PhoneService'])
    MultipleLines = float(request.form['MultipleLines'])
    InternetService = float(request.form['InternetService'])
    MonthlyCharges = float(request.form['MonthlyCharges'])
    TotalCharges = float(request.form['TotalCharges'])

    model = pickle.load(open('svc_model2.pkl', 'rb'))
    data = [[customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService, MultipleLines, InternetService, MonthlyCharges, TotalCharges]]
    df = pd.DataFrame(data, columns=['customerID', 'gender','SeniorCitizen','Partner','Dependents', 'tenure',
           'PhoneService','MultipleLines','InternetService', 'MonthlyCharges', 'TotalCharges'])

    categorical_feature = {feature for feature in df.columns if df[feature].dtypes == 'O'}

    encoder = LabelEncoder()
    for feature in categorical_feature:
        df[feature] = encoder.fit_transform(df[feature])

    single = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    probability = probability*100

    if single == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"
    else:
        op1 = "This Customer is likely to be Continue!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"

    return render_template("index.html", op1=op1, op2=op2,
                           customerID=request.form['customerID'],
                           gender=request.form['gender'],
                           SeniorCitizen=request.form['SeniorCitizen'],
                           Partner=request.form['Partner'],
                           Dependents=request.form['Dependents'],
                           tenure=request.form['tenure'],
                           PhoneService=request.form['PhoneService'],
                           MultipleLines=request.form['MultipleLines'],
                           InternetService=request.form['InternetService'],
                           MonthlyCharges=request.form['MonthlyCharges'],
                           TotalCharges=request.form['TotalCharges'])


if __name__ == '__main__':
    app.run()