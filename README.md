# 🪶 Fikr-e-Ghalib — A Poetic AI Agent

**Fikr-e-Ghalib** is an AI-powered poetic agent inspired by the legendary Mirza Ghalib. It reads current news, reflects on the emotional essence of each event, and composes a 2–4 line verse in Roman Urdu—just as Ghalib might if he lived in today's world.

## ✨ Features

* 📰 Accepts any news article URL
* 🧠 Summarizes the content with deep reasoning
* 🎭 Extracts a dominant theme and sentiment
* 📜 Generates Ghalib-style verse in Roman Urdu
* 💬 Streamlit-based UI for interactive poetry
* ⚙️ Local model fine-tuned with LoRA on Qwen 1.5B

## 🏗️ Stack

* [Streamlit](https://streamlit.io/) for the interactive UI
* [Ollama](https://ollama.com/) to run local LLMs
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for model integration
* [Newspaper3k](https://newspaper.readthedocs.io/en/latest/) for news parsing
* Fine-tuned on a custom Ghalib + reasoning dataset with 2000+ samples

## 🚀 Usage

### 1. Prerequisites

* Install [Ollama](https://ollama.com/) for your platform
* Pull the DeepSeek R1 base model:
```bash
ollama pull deepseek-r1
```

### 2. Clone & Install

```bash
git clone https://github.com/babanomania/fikr-e-ghalib.git
cd fikr-e-ghalib
pip install -r requirements.txt
```

### 3. Fine-tune the Model

```bash
python lib/model_fine_tune2.py
```

This will:
- Load the base Qwen3 0.6B model
- Apply LoRA adaptations using the Ghalib dataset
- Save the fine-tuned model weights

### 4. Run the Agent with Streamlit

```bash
streamlit run app.py
```

### 5. Paste any article URL and get poetic commentary ✨


## ⚙️ Model Training Summary

* Base model: `Qwen/Qwen3-0.6B-Chat`
* Tuned using LoRA adapters with 2000 samples
* Dataset contains:

  * Theme, sentiment, and “thought process”
  * Poetic response in English
  * Chat-formatted prompts using `<|im_start|>` / `<|im_end|>`

## 📬 License

MIT — poetry is meant to be shared.

---

> **_इल्म की राह की कोई इंतिहा नहीं, ये हक़ीक़त दिल जाने है, ग़ालिब,<br/>
हर फ़तहयाब चोटी दिखाती है मंज़र, उन अनगिनत अनदेखे औराक़ का।_**

_Learning never ends. Ghalib lives on in every thought._
