from flask import Flask, render_template, request

from project.model import Model

app = Flask(__name__)
model = Model()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = [request.form["message"]]
    prediction = model.predict(data)
    return render_template("result.html", prediction=prediction)
