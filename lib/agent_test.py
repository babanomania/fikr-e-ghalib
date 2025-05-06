from lib.agent_lib import ghalib_agent, summarize_article, extract_theme_sentiment, extract_roman_urdu_verse
import logging
import sys

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


if __name__ == "__main__":
    logger.info("Starting Ghalib agent application")
    test_url = "https://sports.ndtv.com/cricket/singer-rahul-vaidhya-mocks-virat-kohli-over-avneet-kaur-row-says-rcb-star-blocked-him-8341220"
    try:
        ghalib_agent(test_url)
        logger.info("Application completed successfully")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)
