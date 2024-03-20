from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

scaler = pickle.load(open("/config/workspace/model/train.pkl", "rb"))
model = pickle.load(open("/config/workspace/model/modelPredict.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':
        ## 'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase','Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio', 'Dataset', 'Gender_encoded'
        #male=int(request.form.get("male"))
        Age = float(request.form.get('Age'))
        #Gender = float(request.form.get('Gender'))
        Total_Bilirubin = float(request.form.get('Total_Bilirubin'))
        Direct_Bilirubin = float(request.form.get('Direct_Bilirubin'))
        Alkaline_Phosphotase = float(request.form.get('Alkaline_Phosphotase'))
        Alamine_Aminotransferase = float(request.form.get('Alamine_Aminotransferase'))
        Aspartate_Aminotransferase = float(request.form.get('Aspartate_Aminotransferase'))
        Total_Protiens = float(request.form.get('Total_Protiens'))
        Albumin = float(request.form.get('Albumin'))
        Albumin_and_Globulin_Ratio = float(request.form.get('Albumin_and_Globulin_Ratio'))
        Gender_encoded = float(request.form.get('Gender_encoded'))
        #diaBP = float(request.form.get('diaBP'))
        #BMI = float(request.form.get('BMI'))
        #heartRate = float(request.form.get('heartRate'))
        #glucose = float(request.form.get('glucose'))

        new_data=scaler.transform([[Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Gender_encoded]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Liver Disease'
        else:
            result ='Non-Liver Disease'
            
        return render_template('home.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")