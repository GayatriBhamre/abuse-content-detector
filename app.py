from flask import Flask, render_template, request, jsonify
import joblib
import re

app = Flask(__name__)
model = joblib.load('text_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get('comment', '')
    comment_cleaned = re.sub(r"[^a-zA-Z\s]", "", comment.lower())
    X = vectorizer.transform([comment_cleaned])
    prediction = int(model.predict(X)[0])
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
