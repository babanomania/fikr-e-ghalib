import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset

#######################
# Device Configuration
#######################
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

#######################
# Model Configuration
#######################
MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "models/fikr-e-ghalib-qwen3-lora"

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    device_map="auto" if device != "cpu" else None,
    trust_remote_code=True,
)

#######################
# LoRA Configuration
#######################
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", 
        "o_proj", "gate_proj", 
        "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

#######################
# Dataset Preparation
#######################
def generate_conversation(example):
    """Convert dataset examples into conversation format"""
    input = f"{example['instruction']}\n{example['input']}"
    output = example["output"].replace("Reasoning:", "<think>").replace("Poem:", "</think>")
    return {
        "conversation": [
            {"role": "user", "content": input},
            {"role": "assistant", "content": output}
        ]
    }

# Load and process dataset
dataset = load_dataset("json", data_files="data/dataset3.jsonl")["train"]
dataset = dataset.map(generate_conversation)
conversations = tokenizer.apply_chat_template(dataset["conversation"], tokenize=False)

# Convert to pandas and shuffle
df = pd.DataFrame({"text": conversations})
combined_dataset = Dataset.from_pandas(df)
combined_dataset = combined_dataset.shuffle(seed=3407)

#######################
# Tokenization
#######################
def tokenize(example):
    """Tokenize the text with padding and truncation"""
    return tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

tokenized_dataset = combined_dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

#######################
# Training Setup
#######################
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=0.2, # Change to 1 for full training
    warmup_steps=5,
    learning_rate=2e-5,
    logging_steps=10,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    save_strategy="epoch",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

#######################
# Training Execution
#######################
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train and save
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")