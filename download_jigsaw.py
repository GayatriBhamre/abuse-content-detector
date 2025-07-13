from datasets import load_dataset
import pandas as pd

# Load Jigsaw dataset (Wikipedia comments with toxicity labels)
ds = load_dataset('thesofakillers/jigsaw-toxic-comment-classification-challenge', split='train')
df = ds.to_pandas()

# Keep only toxic labels (binary)
df = df[['comment_text', 'toxic']]
df.columns = ['comment', 'label']

# Save CSV
df.to_csv('data_jigsaw.csv', index=False)
print("✅ Saved data_jigsaw.csv with", len(df), "examples!")
