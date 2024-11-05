import time
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
import evaluate
import torch

# Load the tokenizer and model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load a small part of the MultiNews dataset for testing
train_dataset = load_dataset("multi_news", split="train[:1%]", trust_remote_code=True)
test_dataset = load_dataset("multi_news", split="test[:1%]", trust_remote_code=True)

# Get actual summaries for ROUGE computation
actual_summaries = test_dataset['summary']

# Tokenization function
def tokenize_function(examples):
    prompt = "Provide an original summary that captures the main ideas"
    inputs_with_prompt = [prompt + doc for doc in examples['document']]
    inputs = tokenizer(inputs_with_prompt, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(examples['summary'], truncation=True, padding="max_length", max_length=128)
    inputs['labels'] = labels['input_ids']
    return inputs



# Timing tokenization
start_time = time.time()
print("Starting tokenization of the training dataset...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
print("Tokenization of the training dataset completed in {:.2f} seconds.".format(time.time() - start_time))

start_time = time.time()
print("Starting tokenization of the test dataset...")
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
print("Tokenization of the test dataset completed in {:.2f} seconds.".format(time.time() - start_time))

# Set the format for PyTorch
tokenized_train_dataset.set_format(type='torch')
tokenized_test_dataset.set_format(type='torch')

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

# Get the PEFT model with LoRA
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir='../../../results',
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Smaller batch size for testing
    per_device_eval_batch_size=2,
    num_train_epochs=2,  # Fewer epochs for testing
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

    # Iterate through the dataset
    for example in dataset:
        # Extract input IDs and attention masks from the tokenized dataset
        input_ids = example['input_ids'].unsqueeze(0).to(device)
        attention_mask = example['attention_mask'].unsqueeze(0).to(device)

        # Generate summary using the model
        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=150)

        # Decode the generated summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(summary)

    print("Summary generation completed.")
    return predictions



# Generate summaries and compute ROUGE scores
predictions = generate_summaries_with_prompts(tokenized_test_dataset)
rouge_scores = rouge.compute(predictions=predictions, references=actual_summaries)
print("ROUGE Scores:", rouge_scores)
