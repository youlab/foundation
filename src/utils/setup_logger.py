import logging
import multiprocessing
import sys
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

from config import DIR_LOGS


def setup_logger(file_name):
    # Set a base formatter (can also do JSON here)

    # Create a global log queue
    log_queue = multiprocessing.Queue()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set up file handler with rotation
    file_handler = RotatingFileHandler(DIR_LOGS / file_name, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    # Optional: also log to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Attach both handlers to a listener (in a background thread)
    listener = QueueListener(log_queue, file_handler, console_handler)
    listener.start()

    # Set up root logger to use the queue handler
    queue_handler = QueueHandler(log_queue)
    # queue_handler = DebugQueueHandler(log_queue)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Remove default handlers
    root_logger.addHandler(queue_handler)

    return listener
