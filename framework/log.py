import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s :: %(message)s')


def logger():
    return logging.getLogger(__name__)
