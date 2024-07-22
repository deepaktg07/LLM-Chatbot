from datasets import load_dataset
from transformers import RagTokenForGeneration, RagTokenizer, Trainer, TrainingArguments

# Load the dataset
dataset = load_dataset('json', data_files={'train': 'train_dataset.json'})

# Define the tokenizer and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

# Define the data collator
def data_collator(features):
    questions = [f["question"] for f in features]
    contexts = [f["context"] for f in features]
    inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(contexts, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    inputs["labels"] = labels
    return inputs

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()
