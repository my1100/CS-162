import argparse
import json
import jsonlines
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
from sklearn.metrics import classification_report

# Load model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("LoRA_model")
base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model = PeftModel.from_pretrained(base_model, "LoRA_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def load_entries(path):
    if path.endswith(".jsonl"):
        with jsonlines.open(path) as reader:
            return [obj for obj in reader if "text" in obj and "label" in obj]
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return [obj for obj in json.load(f) if "text" in obj and "label" in obj]
    else:
        raise ValueError("Invalid file")

def predict(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1)
    return preds.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .json or .jsonl file to evaluate")
    args = parser.parse_args()

    entries = load_entries(args.input)
    texts = [e["text"] for e in entries]
    labels = [e["label"] for e in entries]

    if not texts:
        exit(1)

    preds = predict(texts)
    print(classification_report(labels, preds, target_names=["AI-generated", "Human"], digits=4))
