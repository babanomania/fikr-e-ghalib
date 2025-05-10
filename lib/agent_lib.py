from langchain_ollama import OllamaLLM
from newspaper import Article
import re
import logging
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Ollama DeepSeek model
llmSummarizer = OllamaLLM(model="deepseek-r1")  # downloaded from Ollama
llmGhalib = OllamaLLM(model="fikr-e-ghalib-qwen")  # locally added fine-tuned model in Ollama
logger.info("Initialized Ollama DeepSeek model")

# Replace the model initialization section with PEFT model setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
base_model_id = "Qwen/Qwen3-0.6B"
adapter_path = "models/fikr-e-ghalib-qwen3-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(device)
model.eval()
logger.info("Initialized Qwen model and tokenizer with PEFT")

# Step 1: Article summarization function
def summarize_article(url):
    logger.info(f"Starting article summarization for URL: {url}")
    try:
        article = Article(url)
        article.download()
        article.parse()
        content = article.text
        logger.debug(f"Successfully downloaded and parsed article from {url}: {content[:100]}...")
        if not content:
            logger.error("No content found in the article")
            raise ValueError("No content found in the article")

        summary_prompt = f"Summarize the following article concisely:\n\n{content}\n\nSummary:"
        summary = llmSummarizer.invoke(summary_prompt)
        logger.info("Successfully generated article summary")
        return summary.strip()
    except Exception as e:
        logger.error(f"Error in summarize_article: {str(e)}")
        raise

# Step 2: Sentiment & Theme extraction function
def extract_theme_sentiment(summary):
    logger.info("Starting theme and sentiment extraction")
    try:
        extraction_prompt = f"""
        Given the article summary below, identify clearly:
        - One-word Theme:
        - Sentiment (choose from hopeful, tragic, melancholic, uplifting, ironic, pensive, fearful, inspiring, angry, somber):

        Article Summary:
        {summary}

        Theme and Sentiment:
        """
        response = llmSummarizer.invoke(extraction_prompt)
        theme_match = re.search(r"Theme:\s*(\w+)", response)
        sentiment_match = re.search(r"Sentiment:\s*(\w+)", response)

        theme = theme_match.group(1) if theme_match else "general"
        sentiment = sentiment_match.group(1) if sentiment_match else "pensive"

        logger.info(f"Extracted theme: {theme}, sentiment: {sentiment}")
        return theme.lower(), sentiment.lower()
    except Exception as e:
        logger.error(f"Error in extract_theme_sentiment: {str(e)}")
        raise

# Step 3: Ghalib-style verse generation
def generate_ghalib_verse(theme, sentiment, summary):
    logger.info(f"Starting verse generation with theme: {theme}, sentiment: {sentiment}")
    try:
        thought = f"Ghalib would view {theme} through a {sentiment} lens, reflecting on the essence of contradiction."
        
        prompt = f"""Generate a poetic verse in Romanized Urdu inspired by Mirza Ghalib, based on the given theme and sentiment. Keep it authentic to Ghalib's style.
Theme: {theme}
Sentiment: {sentiment}
Thought Process: {thought}
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

        verse = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Raw output from model: {verse}")

        verse = extract_roman_urdu_verse(verse)
        
        logger.info("Successfully generated Ghalib-style verse")
        return verse
    except Exception as e:
        logger.error(f"Error in generate_ghalib_verse: {str(e)}")
        raise


def extract_roman_urdu_verse(raw_output: str) -> str:
    
    # Remove everything before 'assistant'
    assistant_block = raw_output.split("assistant", 1)[-1].strip()

    # Print thinking process if present
    think_matches = re.findall(r'<think>(.*?)</think>', assistant_block, re.DOTALL)
    if think_matches:
        print("\n💭 Thinking Process:")
        for thought in think_matches:
            print(thought.strip())

    # Remove think tags and their content
    cleaned_output = re.sub(r'<think>.*?</think>', '', assistant_block, flags=re.DOTALL)

    # Remove any non-Latin Unicode chunks (Chinese, etc.)
    latin_only = re.split(r"[^\x00-\x7F]+", cleaned_output)[0].strip()

    # Optional: normalize repetitive lines
    lines = latin_only.splitlines()
    unique_lines = []
    for line in lines:
        if line.strip() and (line.strip() not in unique_lines):
            unique_lines.append(line.strip())

    return "\n".join(unique_lines)

# Complete agent workflow
def ghalib_agent(url):
    logger.info(f"Starting Ghalib agent workflow for URL: {url}")
    try:
        summary = summarize_article(url)
        theme, sentiment = extract_theme_sentiment(summary)
        verse = generate_ghalib_verse(theme, sentiment, summary)

        print(f"📰 URL: {url}\n")
        print(f"📌 Summary: {summary}\n")
        print(f"🎭 Theme: {theme.capitalize()}, Sentiment: {sentiment.capitalize()}\n")
        print("✨ Ghalib-Style Verse:")
        print(verse)
        
        logger.info("Successfully completed Ghalib agent workflow")
        return verse
    except Exception as e:
        logger.error(f"Error in ghalib_agent workflow: {str(e)}")
        raise
