import pandas as pd
import json

# Load CSV
df = pd.read_csv('AI_Human.csv')

# Open a .jsonl file to write
with open('kaggle.jsonl', 'w') as f:
    for _, row in df.iterrows():
        json_obj = {"text": row['text'], "generated": row['generated']}
        f.write(json.dumps(json_obj) + '\n')