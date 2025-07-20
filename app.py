from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
model, vectorizer = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    polite_suggestion = None
    comment = ""

    if request.method == "POST":
        comment = request.form["comment"]
        if comment.strip() != "":
            vectorized = vectorizer.transform([comment])
            prediction_label = model.predict(vectorized)[0]
            prediction_proba = model.predict_proba(vectorized)[0]
            confidence_score = np.max(prediction_proba) * 100

            prediction = "Abusive" if prediction_label == 1 else "Clean"
            confidence = f"{confidence_score:.2f}%"

            if prediction == "Abusive":
                polite_suggestion = "Try saying this instead: ‘I didn’t like that behavior.’"

    return render_template("index.html", 
                           prediction=prediction, 
                           confidence=confidence, 
                           comment=comment, 
                           polite_suggestion=polite_suggestion)

if __name__ == "__main__":
    app.run(debug=True)
