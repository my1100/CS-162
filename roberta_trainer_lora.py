from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import torch
import os
import json
from datasets import load_dataset, Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

# Step 1: Load and save tokenized dev set
tokenized_train_path = "tokenized_train_hc3"
tokenized_dev_path = "tokenized_dev"
dev_dataset = None

if os.path.exists(tokenized_dev_path):
    print("Loading tokenized dev dataset from disk...")
    dev_dataset = load_from_disk(tokenized_dev_path)
else:
    print("Tokenizing dev dataset from scratch...")
    dev_entries = []
    dev_folder = "/home/marie/dev-data"

    for filename in os.listdir(dev_folder):
        if filename.endswith(".jsonl"):
            with open(os.path.join(dev_folder, filename), "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("human_text", "").strip():
                        dev_entries.append({"text": data["human_text"].strip(), "label": 1})
                    if data.get("machine_text", "").strip():
                        dev_entries.append({"text": data["machine_text"].strip(), "label": 0})

    dev_dataset = Dataset.from_list(dev_entries)
    dev_dataset = dev_dataset.map(tokenize, batched=True, num_proc=4)
    dev_dataset = dev_dataset.rename_column("label", "labels")
    dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset.save_to_disk(tokenized_dev_path)
    print("Dev dataset tokenized and saved.")

# Step 2: Load and save tokenized training set
# tokenized_train_path = "tokenized_train"
# train_dataset = None
#
# if os.path.exists(tokenized_train_path):
#     print("Loading tokenized train dataset from disk...")
#     train_dataset = load_from_disk(tokenized_train_path)
# else:
#     print("Loading and tokenizing training dataset...")
#     train_dataset = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset", split="train")
#
#     def format_example(example):
#         return {
#             "text": example["data"],
#             "label": 0 if example["generated"] else 1
#         }
#
#     train_dataset = train_dataset.map(format_example)
#     train_dataset = train_dataset.map(tokenize, batched=True, num_proc=4)
#     train_dataset = train_dataset.rename_column("label", "labels")
#     train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
#     train_dataset.save_to_disk(tokenized_train_path)
#     print("Train dataset tokenized and saved.")

train_dataset = load_from_disk(tokenized_train_path)

# Step 3: Load base model
print("Loading base RoBERTa model...")
base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Step 4: Configure LoRA
print("Wrapping model with LoRA adapters...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"], 
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(base_model, lora_config)

# Step 5: Instantiate training arguments
output_dir = "./lora-roberta-checkpoints"
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    label_names=["labels"], 
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    resume_from_checkpoint=True if os.path.exists(output_dir) else False
)

# Step 6: Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# Step 7: Instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

# Step 8: Train model
from pathlib import Path

print("Starting (or resuming) training...")
# Check if a checkpoint exists
checkpoints = list(Path(output_dir).glob("checkpoint-*"))
if checkpoints:
    latest_checkpoint = str(sorted(checkpoints, key=lambda x: int(x.name.split("-")[-1]))[-1])
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("Starting training from scratch.")
    trainer.train()

# Step 9: Evaluate model
print("Evaluating model on dev set...")
metrics = trainer.evaluate()
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Step 10: Save model
model = model.to("cpu")  # Make sure all tensors are loaded and on CPU
model.save_pretrained("./my_model_lora_hc3")
tokenizer.save_pretrained("./my_model_lora_hc3")
print("Training complete. Model + tokenizer saved to './my_model_lora_hc3'.")
