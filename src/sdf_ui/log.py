__docformat__ = "google"

import logging

_LOGGER = logging.getLogger("sdf_ui")
_LOGGER.addHandler(logging.NullHandler())


def logger():
    """
    Returns the logger for the current module.

    Returns:
    logging.Logger: The logger.

    Example:
    >>> log = logger()
    >>> log.debug("Debug message")
    """
    return _LOGGER
