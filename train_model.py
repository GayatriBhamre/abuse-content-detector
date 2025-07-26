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
        "I hate you so much.",
        "You're a wonderful friend.",
        "You disgust me!",
        "I'm really proud of you.",
        "You're a piece of trash.",
        "You always help others.",
        "Get lost, loser!",
        "That's very thoughtful of you.",
        "You're an annoying pest.",
        "I appreciate your efforts.",
        "You make me sick.",
        "Thanks for being there!",
        "You're dumb as a rock.",
        "Well done, keep it up!",
        "You're the worst human ever.",
        "That’s very sweet of you!",
        "Screw you and your face.",
        "You're doing amazing work!",
        "Nobody wants you here.",
        "I respect your honesty.",
        "You're a worthless jerk!"
    ],
    "label": [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ]
}


df = pd.DataFrame(data)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["comment"])
y = df["label"]


model = LogisticRegression()
model.fit(X, y)


joblib.dump((model, vectorizer), "model.pkl")
print("✅ model.pkl saved successfully!")

