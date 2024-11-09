import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import flask 
from flask import Flask, request, app, jsonify, url_for, render_template
from flask import Response
from flask_cors import CORS


def model_creation():
    df = pd.read_csv("AirfoilSelfNoise.csv",header=None)
    df.columns= ["Frequency","Angle of Attack","Chord Length","Free-stream velocity","Suction side","Scaled Sound Pressure Level"]
    # print(df.isnull().sum())
    X = df.iloc[:,:-1]
    print(X)
    Y=df.iloc[:,-1]
    print(Y)
    # print (df)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    pickle.dump(regressor,open("model.pkl","wb")) #model is ready

# model_creation()

app = Flask(__name__, template_folder='templates')
model=pickle.load(open("model.pkl","rb"))

# @app.route('/predict_api',methods=['POST'])
# def predict():
#     data=request.json['data']
#     print(data)
#     new_data = [list(data.values())]
#     output=model.predict(new_data)[0]
#     return jsonify(output)

@app.route('/')
def home():
    return render_template('airfoil.html')

@app.route('/predict',methods=["POST"])
def predict():
    data=[float(x) for x in request.form.values()]
    find_features = [np.array(data)]
    print(find_features)
    output = model.predict(find_features)[0]
    return render_template('airfoil.html',prediction_text="Air Foil Pressure is {}".format(output))


if (__name__)=="__main__":
    app.run(debug=True)