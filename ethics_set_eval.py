import os
import json
import torch
import warnings
from tqdm import tqdm
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    logging as hf_logging
)
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report
)

# Suppress warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", 
                        category=UserWarning)
warnings.filterwarnings("ignore", 
                        category=RuntimeWarning)

# Load model and tokenizer
print("Loading tokenizer and LoRA model...")
tokenizer = RobertaTokenizer.from_pretrained("my_model_lora_hc3")
base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", 
                                                              num_labels=2)
model = PeftModel.from_pretrained(base_model, "my_model_lora_hc3").to("cpu").eval()
print("Model and tokenizer loaded.")

# File reading utilities
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_entries(entries, 
                    human_key="human_text", 
                    machine_key="machine_text"):
    data = []
    for e in entries:
        if human_key in e and e[human_key].strip():
            data.append({"text": e[human_key].strip(), 
                         "label": 1}) 
        if machine_key in e and e[machine_key].strip():
            data.append({"text": e[machine_key].strip(), 
                         "label": 0}) 
    return data

# Evaluation function
def evaluate_dataset(name, entries):
    texts = [ex["text"] for ex in entries]
    true_labels = [ex["label"] for ex in entries]

    encodings = tokenizer(texts, 
                          truncation=True, 
                          padding=True, 
                          max_length=256, 
                          return_tensors="pt")
    preds = []

    for i in tqdm(range(0, len(texts), 8), 
                  desc=f"Evaluating {name}"):
        input_ids = encodings["input_ids"][i:i+8]
        attention_mask = encodings["attention_mask"][i:i+8]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask)
            batch_preds = outputs.logits.argmax(dim=1).tolist()
            preds.extend(batch_preds)

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    print(f"{name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}, N = {len(preds)}")
    print()
    print("Classification Report:")
    print(classification_report(true_labels, preds, target_names=["Human", "AI"], digits=4))

    return {"accuracy": acc, 
            "f1": f1, 
            "n_samples": len(preds)}

# Evaluate ethics set
results = {}
for filename in {"german_wikipedia.jsonl", 
                 "hewlett.json", 
                 "toefl.json"}:
    path = os.path.join("testing_data", 
                        filename)
    
    if filename.endswith(".jsonl"):
        entries = read_jsonl(path)
        examples = prepare_entries(entries)
    elif filename.endswith(".json"):
        entries = read_json(path)
        examples = [{"text": e["document"], "label": 1} for e in entries]

    if examples:
        print()
        print(f"Evaluating file: {filename}")
        results[filename] = evaluate_dataset(filename, examples)
