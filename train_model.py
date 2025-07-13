import pandas as pd
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

print(" Loading Jigsaw dataset...")
ds = load_dataset('thesofakillers/jigsaw-toxic-comment-classification-challenge', split='train')
real_data = ds.to_pandas()[['comment_text', 'toxic']]
real_data.columns = ['comment', 'label']
real_data['label'] = real_data['label'].astype(int)
real_data = real_data.dropna()

print(" Loaded", len(real_data), "rows.")

# -----------------
# Add MANY custom examples
# -----------------
print(" Adding many copies of custom examples...")
abusive_examples = [
    "You are fool", "You are stupid", "I hate you", "Go to hell",
    "You bastard", "Shut up idiot", "Bloody fool", "Die idiot",
    "You are worthless", "You moron"
] * 5  # 50 copies

clean_examples = [
    "You are amazing", "Have a great day", "I appreciate you",
    "You are smart", "Thank you so much", "You are kind",
    "I like you", "You are wonderful", "Good morning", "Best wishes"
] * 5  # 50 copies

custom_df = pd.DataFrame({
    'comment': abusive_examples + clean_examples,
    'label': [1]*len(abusive_examples) + [0]*len(clean_examples)
})

print(f" Custom examples added: {len(custom_df)} (Abusive: {len(abusive_examples)}, Clean: {len(clean_examples)})")

# Combine real data + custom
combined = pd.concat([real_data, custom_df], ignore_index=True)
print("Combined data size:", len(combined))

# -----------------
# Clean text
# -----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

combined['comment'] = combined['comment'].apply(clean_text)

# -----------------
# Vectorize
# -----------------
print(" Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=5)
X = vectorizer.fit_transform(combined['comment'])
y = combined['label']

print(" Label distribution before SMOTE:", pd.Series(y).value_counts().to_dict())

# -----------------
# SMOTE Balancing
# -----------------
print(" Applying SMOTE...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print(" Label distribution after SMOTE:", pd.Series(y_res).value_counts().to_dict())

# -----------------
# Train Model
# -----------------
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_res, y_res)

# -----------------
# Save Model
# -----------------
joblib.dump(model, 'text_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print(" Model and vectorizer saved successfully!")
