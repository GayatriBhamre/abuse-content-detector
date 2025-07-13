from datasets import load_dataset
import pandas as pd

# Load Davidson's hate speech dataset from Hugging Face
ds = load_dataset('tdavidson/hate_speech_offensive')

# Convert to pandas
df = ds['train'].to_pandas()

# Keep only text and label
df = df[['tweet', 'class']]
df.columns = ['comment', 'label']

# Preview
print(df.head())

# Save to CSV
df.to_csv('data.csv', index=False)
print("✅ data.csv saved with real tweets!")
