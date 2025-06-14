from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from transformers import RobertaTokenizer
import os, json

tokenized_train_path = "tokenized_train_hc3"
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def flatten_hc3(example):
    entries = []
    for answer in example["human_answers"]:
        entries.append({"text": answer, "label": 1})
    for answer in example["chatgpt_answers"]:
        entries.append({"text": answer, "label": 0})
    return entries

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = load_dataset("Hello-SimpleAI/HC3", name="all", split="train")

flattened_data = []
for ex in train_dataset:
    flattened_data.extend(flatten_hc3(ex))
train_dataset = Dataset.from_list(flattened_data)

# Check if partial tokenized dataset exists
if os.path.exists(tokenized_train_path):
    try:
        print("Attempting to load partial tokenized data")
        tokenized_existing = load_from_disk(tokenized_train_path)
        processed_ids = set(tokenized_existing["input_ids"])
        print(f"Loaded {len(tokenized_existing)} already-tokenized samples.")
    except Exception as e:
        print(f"Partial load failed: {e}")
        tokenized_existing = None
        processed_ids = set()
else:
    tokenized_existing = None
    processed_ids = set()

# Filter pre-tokenized entries
if processed_ids:
    print("Filtering already tokenized entries...")
    train_dataset = train_dataset.filter(lambda x: tokenizer(x["text"])["input_ids"] not in processed_ids)

# Tokenize data
if len(train_dataset) > 0:
    print(f"Tokenizing remaining {len(train_dataset)} samples")
    train_dataset = train_dataset.map(tokenize, batched=True, num_proc=4)
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Merge with existing and save
    if tokenized_existing:
        train_dataset = concatenate_datasets([tokenized_existing, train_dataset])

    print("Saving tokenized training set to disk")
    train_dataset.save_to_disk(tokenized_train_path)
else:
    print("No new samples to tokenize")
