# CS-162: AI-Generated Text Detection with SVM and LoRA-Finetuned RoBERTa
This project detects AI-generated text using both a traditional Support Vector Machine (SVM) baseline and a transformer-based RoBERTa model fine-tuned with Low-Rank Adaptation (LoRA). Models are trained and evaluated on public datasets with both human and AI-generated content.

---

## Features

- SVM baseline (linear kernel, statistical features)
- LoRA-finetuned RoBERTa transformer model
- Training and evaluation on HC3, Reddit, and other public corpora
- Comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC
- Error analysis and cross-domain robustness

---

## Datasets

- [HC3 (Human ChatGPT Comparison Corpus)](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese): English Q&A pairs from ELI5, finance, medicine, open-domain, and CS/AI.
- [ahmadreza13 Human-vs-AI Dataset](https://huggingface.co/datasets/ahmadreza13/human-vs-Ai-generated-dataset): 3.6M samples.
- Evaluation sets: `arxiv_chatgpt`, `arxiv_cohere`, `reddit_chatgpt`, `reddit_cohere`, `german_wikipedia`, `toefl`, `hewlett`

---

## Modeling Approach

- SVM: linear kernel, feature-based baseline
- RoBERTa + LoRA: encoder-only transformer with LoRA adapters

## Test the Model

To run the model, clone the git repository and open up a terminal in the folder.

Then, run the following:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This will set up the environment.

Then, to evaluate a specific .json or .jsonl file, run the following:

```
python grade_model.py --input [name of file to test] 
```