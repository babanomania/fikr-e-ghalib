from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load the model and tokenizer
base_model_id = "Qwen/Qwen3-0.6B"
adapter_path = "models/fikr-e-ghalib-qwen3-lora"

# Load the PEFT model
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", trust_remote_code=True)

# Load the PEFT model
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(device)
model.eval()

# Generate a poetic couplet
prompt = """Generate a poetic couplet in English inspired by Mirza Ghalib, based on the given theme and sentiment. Provide reasoning.
Theme: betrayal
Sentiment: pensive
"""

messages = [{"role": "user", "content": prompt}]

chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_input, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
