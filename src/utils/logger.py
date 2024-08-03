# src/utils/logger.py

from loguru import logger
import sys

# Configure loguru
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add a handler for console output
logger.add("logs/file_{time}.log", rotation="500 MB")  # Add a handler for file output

# You can add more custom configurations here if needed

# Export the logger
log = logger