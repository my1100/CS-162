import pandas as pd
import json

# Load CSV
df = pd.read_csv('AI_Human.csv')

# Open a .jsonl file to write
with open('kaggle.jsonl', 'w') as f:
    for _, row in df.iterrows():
        if row['generated'] == 0:
            json_obj = {"human_text": row['text']}
        elif row['generated'] == 1:
            json_obj = {"machine_text": row['text']}
        else:
            continue  # skip invalid values
        f.write(json.dumps(json_obj) + '\n')