from langchain_ollama import OllamaLLM
from newspaper import Article
import re
import logging
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# Initialize the model and tokenizer
model_path = "models/fikr_e_ghalib_qwen_lora"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float32)
tokenizer.pad_token = tokenizer.eos_token
logger.info("Initialized Qwen model and tokenizer")

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
        
        prompt = f"""<|im_start|>system
                You are a poetic AI trained to think like Mirza Ghalib and write couplets only in Romanized Urdu using the Latin script.
                <|im_end|>
                <|im_start|>user
                Theme: {theme}
                Sentiment: {sentiment}
                Thought Process: {thought}
                Generate a 2-3 line poetic verse in Romanized Urdu inspired by Mirza Ghalib.
                <|im_end|>
                <|im_start|>assistant
                """

        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
            eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        verse = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        
        logger.info("Successfully generated Ghalib-style verse")
        return verse
    except Exception as e:
        logger.error(f"Error in generate_ghalib_verse: {str(e)}")
        raise


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
