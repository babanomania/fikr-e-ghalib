from agent_lib import ghalib_agent
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
    test_url = "https://ideas.ted.com/10-real-life-love-stories-thatll-grab-you-by-the-heart-from-storycorps"
    try:
        ghalib_agent(test_url)
        logger.info("Application completed successfully")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)
