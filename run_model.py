# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
# model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")

a_phrase = "This is a test sentence."

# inputs = tokenizer(a_phrase, return_tensors="pt")

# Forward pass, get logit predictions
# outputs = model(**inputs)
# logits = outputs.logits
# Get probabilities
# probs = logits.softmax(dim=1)
# Get predicted class
# predicted_class = probs.argmax(dim=1).item()
# Print the results
# print(f"Logits: {logits}")
# print(f"Probabilities: {probs}")
# print(f"Predicted class: {predicted_class}")

from transformers import pipeline

pipe = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")     

print(pipe(a_phrase))