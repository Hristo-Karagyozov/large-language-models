# Text Summarization with Parameter-Efficient Fine-Tuning (PEFT) and Prompt Engineering

## Overview

This repository contains the code and experiments for the paper *"Efficient Text Summarization through Parameter-Efficient Fine-Tuning and Prompt Engineering"*. The study explores the effectiveness of parameter-efficient fine-tuning (PEFT) methods, specifically **LoRA (Low-Rank Adaptation)**, and **prompt engineering** in improving the performance of smaller, fine-tuned models for abstractive text summarization. 

We evaluate multiple model configurations, including:
- A **PEFT fine-tuned BART model**
- A **fully fine-tuned BART model**
- A **non-fine-tuned T5 model**

The models are tested on the **MultiNews dataset**, a challenging dataset for abstractive summarization.

## Key Results

- **PEFT fine-tuned models**: Smaller models using LoRA can achieve competitive summarization performance while significantly reducing computational costs.
- **Prompt Engineering**: The use of task-specific prompts further enhances summarization quality.
- **Resource Efficiency**: Our findings highlight the importance of **parameter-efficient fine-tuning** and **prompt engineering** as strategies to make large pre-trained models more accessible and computationally efficient without sacrificing performance.

## Setup

To reproduce the experiments or run the models, follow the steps below.

### Prerequisites

- Python 3.7+
- PyTorch 1.10+
- Hugging Face `transformers` library
- `datasets` library from Hugging Face
- `evaluate` library for ROUGE metric evaluation




Model Size and Computation Cost: While PEFT significantly reduces the computational cost, the models still require substantial resources for training and fine-tuning. Optimizing for smaller, even more efficient models could reduce the resource consumption further.

Performance of LoRA vs. Full Fine-Tuning: The PEFT fine-tuned models, while efficient, demonstrated a small trade-off in performance compared to fully fine-tuned models. Future improvements in LoRA or alternative PEFT methods could help bridge this gap and achieve even better results.

Dataset Complexity: The MultiNews dataset, being a multi-document summarization task, can sometimes result in challenging and non-intuitive summaries. Future work could explore different datasets with more diverse domains and test the scalability of PEFT techniques across a broader range of tasks.

Model Generalization: While the models perform well on the MultiNews dataset, their ability to generalize to other summarization datasets or even different tasks (e.g., question answering, translation) has yet to be fully explored.

Limited Evaluation Metrics: While ROUGE scores are widely used for evaluating summarization tasks, exploring additional evaluation metrics such as BERTScore or human evaluation would provide a more comprehensive understanding of model performance.
Install the required dependencies:

```bash
pip install -r requirements.txt
