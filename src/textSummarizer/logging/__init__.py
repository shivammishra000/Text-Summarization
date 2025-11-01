import os
import sys
import logging

# Log format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Log file path
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger instance
logger = logging.getLogger("textSummarizerLogger")

# Prevent duplicate handlers if imported multiple times
if logger.hasHandlers():
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_filepath))
    logger.addHandler(logging.StreamHandler(sys.stdout))
