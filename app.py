# import relevant libraries for flask,html rendering and loading the ML model
from flask import Flask, request, url_for, redirect, render_template
import pickle
import pandas as pd
import joblib

app = Flask(__name__)

# model = pickle.load(open("model.pkl","rb"))
model = joblib.load("model.pkl")
# scale = pickle.load(open("scale.pkl","rb"))
scale = joblib.load("scale.pkl")


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():

    pregnancies = request.form['1']
    glucose = request.form['2']
    bloodPressure = request.form['3']
    skinThickness = request.form['4']
    insulin = request.form['5']
    bmi = request.form['6']
    dpf = request.form['7']
    age = request.form['8']

    rowDF = pd.DataFrame([pd.Series([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age])])
    rowDF_new = pd.DataFrame(scale.transform(rowDF))

    print(rowDF_new)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
