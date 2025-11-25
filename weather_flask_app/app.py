from flask import Flask, render_template, request
import pickle
import numpy as np

# Creating the Flask app
app = Flask(__name__)

# Loading the trained model
with open("weather_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Getting form values and converting to float
            precipitation = float(request.form.get("precipitation"))
            temp_max = float(request.form.get("temp_max"))
            temp_min = float(request.form.get("temp_min"))
            wind = float(request.form.get("wind"))

            # Building the feature array in the SAME order as training:
            # ['precipitation', 'temp_max', 'temp_min', 'wind']
            features = np.array([[precipitation, temp_max, temp_min, wind]])

            # Make prediction
            pred_class = model.predict(features)[0]

            prediction = f"Predicted weather: {pred_class}"

        except ValueError:
            error = "Please enter valid numeric values for all fields."

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    # Run the app locally
    app.run(debug=True)
