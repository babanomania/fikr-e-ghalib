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
    test_url = "https://edition.cnn.com/2025/05/10/politics/habeas-corpus-explained"
    try:
        ghalib_agent(test_url)
        logger.info("Application completed successfully")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)
