import logging

import rich.logging


def create_logger(name, level=logging.INFO, show_time=True, show_level=True):
    """Creates and configures a logger with its own RichHandler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ✅ Create a new RichHandler for this logger
    handler = rich.logging.RichHandler(show_time=show_time, show_level=show_level, show_path=False)

    # ✅ Define a custom formatter
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s" if show_level else "%(message)s")
    handler.setFormatter(formatter)

    # ✅ Remove old handlers before adding a new one (allows redefining later)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)
    return logger
