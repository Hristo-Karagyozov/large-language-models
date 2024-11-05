from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import json
import evaluate
import time

# Load the tokenizer and model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load a small part of the MultiNews dataset for testing
train_dataset = load_dataset("multi_news", split="train[:20%]", trust_remote_code=True)  # 1% for testing
test_dataset = load_dataset("multi_news", split="test[:10%]", trust_remote_code=True)    # 1% for testing

# Tokenization function
def tokenize_function_with_prompt(examples):
    # Adding a prompt to the input text
    inputs_with_prompt = ["Provide an original summary that captures the main ideas " + doc for doc in examples['document']]
    inputs = tokenizer(inputs_with_prompt, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(examples['summary'], truncation=True, padding="max_length", max_length=128)
    inputs['labels'] = labels['input_ids']
    return inputs


# Timing tokenization
start_time = time.time()
print("Starting tokenization of the training dataset...")
tokenized_train_dataset = train_dataset.map(tokenize_function_with_prompt, batched=True, remove_columns=train_dataset.column_names)
print("Tokenization of the training dataset completed in {:.2f} seconds.".format(time.time() - start_time))

start_time = time.time()
print("Starting tokenization of the test dataset...")
tokenized_test_dataset = test_dataset.map(tokenize_function_with_prompt, batched=True, remove_columns=test_dataset.column_names)
print("Tokenization of the test dataset completed in {:.2f} seconds.".format(time.time() - start_time))

# Set the format for PyTorch
tokenized_train_dataset.set_format(type='torch')
tokenized_test_dataset.set_format(type='torch')

# Define training arguments for full fine-tuning
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

# Timing model training
start_time = time.time()
print("Starting model training...")
trainer.train()
print("Model training completed in {:.2f} seconds.".format(time.time() - start_time))

# Load ROUGE evaluation metric
rouge = evaluate.load("rouge")

# Load the model onto the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate summaries
def generate_summaries_with_prompts(dataset):
    print("Starting summary generation with prompts...")
    model.eval()
    predictions = []

    for example in dataset:
        # Recreate the prompt used during training
        input_text = "Provide an original summary that captures the main ideas " + tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=150)

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(summary)

    print("Summary generation completed.")
    return predictions


# Generate summaries and compute ROUGE scores
predictions = generate_summaries_with_prompts(tokenized_test_dataset)
actual_summaries = test_dataset['summary']
rouge_scores = rouge.compute(predictions=predictions, references=actual_summaries)
print("ROUGE Scores:", rouge_scores)

output_path = './results/rouge_scores.json'  # Define the path to save results
with open(output_path, 'w') as f:
    json.dump(rouge_scores, f)
print(f"ROUGE Scores saved to {output_path}")
