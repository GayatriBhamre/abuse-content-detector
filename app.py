from flask import Flask, render_template, request
import joblib
import numpy as np
import random

app = Flask(__name__)


model, vectorizer = joblib.load("model.pkl")

# Polite suggestion list
polite_suggestions_list = [
    "Try saying: 'That wasn’t appropriate.'",
    "Try: 'I found that offensive.'",
    "You could say: 'Please be respectful.'",
    "How about: 'I didn’t like that behavior.'"
     "Try saying: 'That wasn’t appropriate.'",
    "Try: 'I found that offensive.'",
    "You could say: 'Please be respectful.'",
    "How about: 'I didn’t like that behavior.'",
    "You could say: 'Let's keep the conversation kind.'",
    "Try: 'Please watch your language.'",
    "Consider saying: 'That came across as hurtful.'",
    "Try this: 'Let’s speak with kindness.'",
    "You might say: 'That could be phrased more gently.'",
    "How about: 'Let’s be more considerate.'",
    "Try: 'I prefer respectful discussions.'",
    "Consider: 'Let's maintain a positive tone.'",
    "Say instead: 'Let’s be mindful of our words.'",
    "Try: 'Please avoid saying things like that.'",
    "How about: 'That sounded harsh, maybe rephrase?'"
]

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
                polite_suggestion = random.choice(polite_suggestions_list)

    return render_template("index.html", 
                           prediction=prediction, 
                           confidence=confidence, 
                           comment=comment, 
                           polite_suggestion=polite_suggestion)

if __name__ == "__main__":
    app.run(debug=True)
