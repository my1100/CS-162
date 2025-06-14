# CS-162: AI-Generated Text Detection with SVM and LoRA-Finetuned RoBERTa
This project is an AI-generated text detector that leverages both a classical Support Vector Machine (SVM) baseline and a modern transformer-based model—RoBERTa—fine-tuned with Low-Rank Adaptation (LoRA). The goal is to distinguish between human-written and AI-generated content, using datasets sourced from Reddit, arXiv, and the HC3 Human ChatGPT Comparison Corpus.

Features
SVM Baseline: Classical linear SVM trained on extracted linguistic and statistical features.

LoRA-Finetuned RoBERTa: Efficient parameter-efficient fine-tuning of RoBERTa using LoRA for high-accuracy classification.

Dataset Support: Trained primarily on the HC3 dataset, with additional evaluation on several public benchmarks including datasets from Reddit and arXiv.

Comprehensive Evaluation: Detailed metrics (accuracy, precision, recall, F1, ROC-AUC) reported for all models across multiple domains.

Error Analysis: In-depth analysis of failure cases and model biases, including out-of-domain and non-English evaluation.

Datasets
HC3 (Human ChatGPT Comparison Corpus): Contains parallel human and ChatGPT responses to the same prompts, covering a range of topics.

ahmadreza13 Human-vs-AI Dataset: Large-scale (3.6M samples) dataset from HuggingFace.

arxiv_chatgpt, arxiv_cohere, reddit_chatgpt, reddit_cohere, german_wikipedia, toefl, hewlett: Evaluation sets for measuring robustness and generalization.

Modeling
SVM: Linear kernel, classical feature-based approach.

RoBERTa + LoRA: Encoder-only architecture with LoRA adapters (r=8, lora_alpha=16, target modules: “query”, “value”), batch size 8, 3 epochs, weight decay 0.01.

