# src/utils/logger.py

from loguru import logger
import sys
import pprint
import json
import re
from loguru._colorizer import Colorizer

# Monkey-patch the Colorizer._parse_without_formatting method
original_parse_without_formatting = Colorizer._parse_without_formatting


@staticmethod
def patched_parse_without_formatting(string, *, recursion_depth=1000, recursive=False):
    return original_parse_without_formatting(string, recursion_depth=recursion_depth, recursive=recursive)


Colorizer._parse_without_formatting = patched_parse_without_formatting

# Create a custom PrettyPrinter instance
pp = pprint.PrettyPrinter(indent=2, width=100, depth=None, compact=False)


def escape_curly_braces(s):
    return s.replace("{", "{{").replace("}", "}}")


def safe_format(obj):
    """Safely format an object, handling potential recursion issues and preserving actual newlines."""
    try:
        # Try to format using pprint
        formatted = pp.pformat(obj)
    except (ValueError, RecursionError):
        try:
            # If pprint fails, try json dumps
            formatted = json.dumps(obj, default=str, indent=2)
        except:
            # If all else fails, use basic string representation
            formatted = str(obj)

    # Preserve actual newlines and escape curly braces
    # formatted = preserve_newlines(formatted)
    formatted = formatted.replace("<", "\<").replace(">", "\>")
    return escape_curly_braces(formatted)


def format_record(record):
    # Safely format the message
    formatted_message = safe_format(record["message"])

    # Create the log message with the custom format
    log_message = (
        f"<green>{record['time']:YYYY-MM-DD HH:mm:ss}</green> | "
        f"<level>{record['level']:<8}</level> | "
        f"<cyan>{record['name']}</cyan>:<cyan>{record['function']}</cyan>:<cyan>{record['line']}</cyan> - "
        f"<level>{formatted_message}</level>\n"
    )

    return log_message


# Remove the default handler
logger.remove()

# Add a new handler with the custom format
logger.add(
    sys.stderr,
    format=format_record,
    level="INFO",
    colorize=True
)

# If you're also logging to a file, add another handler
logger.add(
    "logs/file_{time}.log",
    format=format_record,
    level="INFO",
    rotation="500 MB",
    colorize=False
)

# Export the logger
log = logger
