from flask import Flask, render_template, request
import pickle
import numpy as np
import streamlit as st

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    product_map = {
        0: "Electronics",
        1: "Fashion",
        2: "Groceries",
        3: "Home Appliances",
        4: "Books"
    }

    if request.method == "POST":
        features = [
            int(request.form["age"]),
            int(request.form["gender"]),
            int(request.form["income"]),
            int(request.form["spending"]),
            int(request.form["browsing"]),
            int(request.form["purchases"]),
            int(request.form["last_category"])
        ]

        scaled_features = scaler.transform([features])
        pred_code = model.predict(scaled_features)[0]
        prediction = product_map[pred_code]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
