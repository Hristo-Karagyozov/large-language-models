from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate
from datasets import load_dataset
from tqdm import tqdm

# Load T5-Large model and tokenizer
model_name = "t5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load MultiNews dataset
dataset = load_dataset("multi_news", split="test[:10%]", trust_remote_code=True)
rouge = evaluate.load("rouge")


# Helper function to generate summaries and evaluate
def generate_summary(batch):
    inputs = tokenizer(batch["document"], return_tensors="pt", padding="max_length", truncation=True,
                       max_length=512)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# Testing loop
references, predictions = [], []
for example in tqdm(dataset, desc="Processing Examples", total=len(dataset)):
    summary = generate_summary(example)
    predictions.append(summary)
    references.append(example["summary"])

    # Print progress occasionally
    if len(predictions) % 100 == 0:
        print(f"Processed {len(predictions)} summaries...")

# Compute ROUGE scores
rouge_scores = rouge.compute(predictions=predictions, references=references)
print("ROUGE Scores:", rouge_scores)
