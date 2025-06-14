import argparse
import json
import jsonlines
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification, logging as hf_logging
from peft import PeftModel
from sklearn.metrics import classification_report

# Suppress warnings
hf_logging.set_verbosity_error()

# Load tokenizer and model
print("Loading tokenizer and LoRA model...")
tokenizer = RobertaTokenizer.from_pretrained("LoRA_model")
base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model = PeftModel.from_pretrained(base_model, "LoRA_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Load entries
def load_entries(path):
    entries = []
    if path.endswith(".jsonl"):
        with jsonlines.open(path) as reader:
            for obj in reader:
                if "text" in obj and "label" in obj:
                    entries.append(obj)
                elif "human_text" in obj:
                    entries.append({"text": obj["human_text"].strip(), "label": 1})
                elif "machine_text" in obj:
                    entries.append({"text": obj["machine_text"].strip(), "label": 0})
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for obj in raw:
                if "text" in obj and "label" in obj:
                    entries.append(obj)
                elif "human_text" in obj:
                    entries.append({"text": obj["human_text"].strip(), "label": 1})
                elif "machine_text" in obj:
                    entries.append({"text": obj["machine_text"].strip(), "label": 0})
                elif "document" in obj: 
                    entries.append({"text": obj["document"].strip(), "label": 1})
    else:
        raise ValueError(f"Unsupported file format")
    return entries

# Make predictions
def predict(texts, batch_size=8):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend(batch_preds)
    return preds

# Execute
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .json or .jsonl file to evaluate")
    args = parser.parse_args()

    entries = load_entries(args.input)
    if not entries:
        print("No valid entries found.")
        exit(1)

    texts = [e["text"] for e in entries]
    labels = [e["label"] for e in entries]

    print(f"\nEvaluating {len(texts)} samples from {args.input}...\n")
    preds = predict(texts)

    print("Classification Report:\n")
    print(classification_report(labels, preds, target_names=["AI-generated", "Human"], digits=4))
