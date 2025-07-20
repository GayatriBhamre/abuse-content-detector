import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


data = {
    "comment": [
        "You are amazing!",
        "You're such an idiot!",
        "Thank you so much!",
        "Shut up, no one cares.",
        "Great job!",
        "Go to hell!",
        "You're the best!",
        "Stupid moron!",
        "You're kind and helpful.",
        "I hate you so much."
    ],
    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = abusive, 0 = clean
}

df = pd.DataFrame(data)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["comment"])
y = df["label"]


model = LogisticRegression()
model.fit(X, y)


joblib.dump((model, vectorizer), "model.pkl")
print("✅ model.pkl saved successfully!")

