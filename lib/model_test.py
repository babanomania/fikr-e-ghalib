from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

def extract_roman_urdu_verse(raw_output: str) -> str:
    # Remove everything before 'assistant'
    assistant_block = raw_output.split("assistant", 1)[-1].strip()

    # Remove any non-Latin Unicode chunks (Chinese, etc.)
    latin_only = re.split(r"[^\x00-\x7F]+", assistant_block)[0].strip()

    # Optional: normalize repetitive lines if needed
    lines = latin_only.splitlines()
    unique_lines = []
    for line in lines:
        if line.strip() and (line.strip() not in unique_lines):
            unique_lines.append(line.strip())

    return "\n".join(unique_lines)


# Load the fine-tuned model
model_path = "./fikr_e_ghalib_qwen_lora"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float32)

# Set pad token if not present
tokenizer.pad_token = tokenizer.eos_token

# Theme + sentiment + thinking prompt
theme = "freedom"
sentiment = "hopeful"
thought = "Ghalib would view freedom through a hopeful lens, finding beauty in contradiction."

prompt = f"""<|im_start|>system
You are a poetic AI trained to think like Mirza Ghalib and explain your thoughts and write couplets only in Roman Urdu using the Latin script. End your response with a poetic pause.
<|im_end|>
<|im_start|>user
Theme: {theme}
Sentiment: {sentiment}
Thought Process: {thought}
Generate a 2-3 line poetic verse in Romanized Urdu inspired by Mirza Ghalib.
<|im_end|>
<|im_start|>assistant
"""

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

# Generate response
output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.9,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=1.2,
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
)

# Decode and clean response
decoded = tokenizer.decode(output[0], skip_special_tokens=True)
output_text = decoded.split("<|im_start|>assistant\\n")[-1].split("<|im_end|>")[0].strip()
cleaned = extract_roman_urdu_verse(output_text)

print("\n📝 Verse:\n")
print(cleaned)
