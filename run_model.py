# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")