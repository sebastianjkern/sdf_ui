__docformat__ = "google"

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s :: %(message)s')


def logger():
    """
    Returns the logger for the current module.

    Returns:
    logging.Logger: The logger.

    Example:
    >>> log = logger()
    >>> log.debug("Debug message")
    """
    return logging.getLogger(__name__)
