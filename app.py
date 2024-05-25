from flask import Flask,render_template,url_for,request,jsonify

import joblib
import numpy as np
model = joblib.load("kmeans_model.lb")
scaler = joblib.load("standard_scaler.lb")



app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/input")
def input():
    return render_template("input.html")

@app.route("/prediction",methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        Nitrogen = float(request.form['N'])
        Phosphorus = float(request.form['P'])
        Pottasium = float(request.form['K'])
        Temperature = float(request.form["T"])
        Humidity = float(request.form['H'])
        PH = float(request.form['ph'])
        Rainfall = float(request.form['rain'])

        user_data = np.array([[Nitrogen,Phosphorus,Pottasium,Temperature,Humidity,PH,Rainfall]])

        user_data_transformed  = scaler.transform(user_data)
        
        prediction = model.predict(user_data_transformed)

        prediction_native = int(prediction[0])

        return render_template("prediction.html", output = str(prediction_native))



if __name__ == "__main__":
    app.run(debug=True)
