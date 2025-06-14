import os
import json
import jsonlines
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, auc
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from datasets import Dataset
import random

from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification
from peft import PeftModel

# Load tokenizer and base model
base_model_path = "roberta-base"
lora_model_path = "./my_model_lora_hc3"

tokenizer = RobertaTokenizer.from_pretrained(base_model_path)
base_model = RobertaForSequenceClassification.from_pretrained(base_model_path, num_labels=2)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_model_path)

# Ensure model is on the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Print device info for debugging
print(f"Running on: {device}")

# Classification function
def classify(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    return preds

# Load JSON or JSONL
def load_entries(filepath):
    if filepath.endswith(".jsonl"):
        with jsonlines.open(filepath) as reader:
            return list(reader)
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

import time

def evaluate_file(filepath):
    texts, labels = [], []
    filename = os.path.basename(filepath)

    print(f"\n--- Loading {filename} ---")
    entries = load_entries(filepath)
    for obj in entries:
        if "human_text" in obj:
            texts.append(obj["human_text"])
            labels.append(1)
        if "machine_text" in obj:
            texts.append(obj["machine_text"])
            labels.append(0)

    if not texts or not labels:
        print(f"Skipping {filename} (no valid entries)")
        return None

    print(f"Loaded {len(texts)} samples from {filename}")

    # Start timing
    start_time = time.time()

    # Predict in batches
    batch_size = 8
    preds = []

    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_preds = classify(batch_texts)
        preds.extend(batch_preds)
        print(f"Batch {i+1}/{num_batches} done", end="\r")

    elapsed = time.time() - start_time
    print(f"\nDone predicting {filename} in {elapsed:.2f} seconds.")

    report = classification_report(labels, preds, digits=4, zero_division=1)
    return report

def evaluate_all(test_folder):
    print("\n=== Evaluating Entire Dataset ===")
    all_texts, all_labels = [], []
    test_files = [f for f in os.listdir(test_folder) if f.endswith((".jsonl", ".json"))]

    for filename in tqdm(test_files, desc="Loading all files"):
        path = os.path.join(test_folder, filename)
        entries = load_entries(path)
        for obj in entries:
            if "human_text" in obj:
                all_texts.append(obj["human_text"])
                all_labels.append(1)
            if "machine_text" in obj:
                all_texts.append(obj["machine_text"])
                all_labels.append(0)

    if not all_texts or not all_labels:
        print("No data found.")
        return

    print(f"Total samples loaded: {len(all_texts)}")
    start_time = time.time()

    batch_size = 8
    all_preds = []
    num_batches = (len(all_texts) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_texts = all_texts[start:end]
        batch_preds = classify(batch_texts)
        all_preds.extend(batch_preds)
        print(f"Batch {i+1}/{num_batches} done", end="\r")

    elapsed = time.time() - start_time
    print(f"\nFinished entire dataset in {elapsed:.2f} seconds.")

    report = classification_report(all_labels, all_preds, digits=4, zero_division=1)
    print("\n=== Classification Report for Entire Dataset ===")
    print(report)

    # ROC-AUC Calculation
    print("\nPlotting ROC-AUC Curve...")
    # Get prediction probabilities instead of hard predictions
    all_probs = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_texts = all_texts[start:end]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Get prob of class 1
            all_probs.extend(probs.cpu().numpy())

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    print("ROC curve saved to: roc_curve.png")


def get_prediction(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        pred_label = torch.argmax(probs).item()
        confidence = probs[pred_label].item()

    label_str = "Human" if pred_label == 1 else "AI-generated"
    print(f"Prediction: {label_str} ({pred_label}) | Confidence: {confidence:.4f}")
    return pred_label, confidence


def interpret_prediction(text, label=1, target_class=None):
    print(f"\nAnalyzing text: {text[:60]}...")

    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    input_ids = inputs["input_ids"].to(device).long()  # ensure it's LongTensor
    attention_mask = inputs["attention_mask"].to(device)

    def forward_func(input_ids):
        input_ids = input_ids.long()  # make sure input is long inside forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.softmax(outputs.logits, dim=-1)[:, target_class]

    if target_class is None:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            target_class = pred

    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(
        inputs=input_ids,
        n_steps=30,
        return_convergence_delta=False
    )

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = attributions.detach().cpu().numpy()

    print(f"Predicted class: {target_class} ({'human' if target_class == 1 else 'machine'})")

    print("\nToken importances:")
    for token, score in zip(input_tokens, scores):
        print(f"{token:<12} {score:.4f}")



# # 1. Evaluate on all dev datasets
# test_folder = "../dev-data"
# evaluate_all(test_folder)

# # 2. Evaluate on individual datasets
# test_files = [f for f in os.listdir(test_folder) if f.endswith((".jsonl", ".json"))]

# for filename in tqdm(test_files, desc="Evaluating all files"):
#     path = os.path.join(test_folder, filename)
#     report = evaluate_file(path)
#     if report is None:
#         continue
#     print(f"\n=== Classification Report for {filename} ===")
#     print(report)

# 3. Pick correct and wrong examples from dev dataset
dev_entries = []
dev_folder = "../dev-data"

for filename in os.listdir(dev_folder):
    if filename.endswith(".jsonl"):
        with open(os.path.join(dev_folder, filename), "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("human_text", "").strip():
                    dev_entries.append({"text": data["human_text"].strip(), "labels": 1})
                if data.get("machine_text", "").strip():
                    dev_entries.append({"text": data["machine_text"].strip(), "labels": 0})

dev_dataset = Dataset.from_list(dev_entries)


correct_preds = []
wrong_preds = []

for example in dev_dataset:
    text = example["text"]
    true_label = example["labels"]
    pred_label, confidence = get_prediction(text)

    result = {
        "text": text,
        "pred": pred_label,
        "true": true_label,
        "confidence": confidence
    }

    if pred_label == true_label and len(correct_preds) < 3:
        correct_preds.append(result)
    elif pred_label != true_label and len(wrong_preds) < 3:
        wrong_preds.append(result)

    if len(correct_preds) == 3 and len(wrong_preds) == 3:
        break

# Print results
print("\n3 Correct Predictions:")
for ex in random.sample(correct_preds, 3):
    print(f"\nText: {ex['text']}...")
    print(f"Prediction: {ex['pred']} | True: {ex['true']} | Confidence: {ex['confidence']:.4f}")

print("\n3 Incorrect Predictions:")
for ex in random.sample(wrong_preds, 3):
    print(f"\nText: {ex['text']}...")
    print(f"Prediction: {ex['pred']} | True: {ex['true']} | Confidence: {ex['confidence']:.4f}")


# # 4. Interpret on individual new examples
print("\nRunning interpretation on example 1...(source: r/ucla)")
sample_text = "Ever since I brought my gaming laptop to school, my grades have gotten better. This is undeniable proof that playing video games is a vital contributor to your academic success. So yes, me playing RDR2 for 6 hours last night does count as studying, thus making me a major academic weapon"
get_prediction(sample_text)

print("\nRunning interpretation on example 1.5...(source: chatgpt)")
sample_text1 = "What’s one underrated spot on campus you wish more people knew about? Been at UCLA for a while now and I feel like I keep discovering random quiet corners or cool views totally by accident. Recently found this little nook near the top floor of the Law Building that overlooks Westwood and barely anyone was there. Super peaceful. Would love to know what hidden gems others have found — could be a study spot, a food hack, a shortcut, or just somewhere to chill. Bonus points if it’s not overcrowded (yet). Also down to make a Google Map of all the spots if there’s interest"
get_prediction(sample_text1)

print("\nRunning interpretation on example 2...(source: LoRA paper abstract)")
sample_text2 = "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA."  # or load from dataset
get_prediction(sample_text2)

print("\nRunning interpretation on example 3...(source: GPT abstract on LoRA)")
sample_text3 = "Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique for large pre-trained models, particularly transformer-based architectures. Instead of updating all model weights during downstream training, LoRA injects trainable low-rank matrices into specific layers (e.g., attention weights), allowing the original weights to remain frozen. This significantly reduces the number of trainable parameters while maintaining performance, making it highly suitable for resource-constrained settings. LoRA has been widely adopted for natural language processing tasks and is especially impactful in scenarios where multiple tasks or domains require fine-tuning of large models."
get_prediction(sample_text3)