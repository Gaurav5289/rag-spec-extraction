import logging
import os
from .config import LOG_DIR

LOG_FILE = os.path.join(LOG_DIR, "system.log")

# Set the file logging level to DEBUG
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger("rag-spec-extraction")

# Also log to console, set to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG) 
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)