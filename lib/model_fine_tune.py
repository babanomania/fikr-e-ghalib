from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# Load Qwen1.5 model and tokenizer
model_name = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float32)

# Apply LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load updated dataset
data_path = "data/dataset.jsonl"
dataset = load_dataset("json", data_files=data_path)["train"]

# Tokenization function
def tokenize(sample):
    input_ids = tokenizer(sample["input"] + sample["output"], truncation=True, max_length=1024)
    input_ids["labels"] = input_ids["input_ids"].copy()
    return input_ids

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Training configuration
training_args = TrainingArguments(
    output_dir="./fikr_e_ghalib_qwen_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    report_to="none"
)

# Set up Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Start fine-tuning
trainer.train()

# Save model
model.save_pretrained("./fikr_e_ghalib_qwen_lora")
tokenizer.save_pretrained("./fikr_e_ghalib_qwen_lora")
