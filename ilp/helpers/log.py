import sys
import logging


def make_logger(name, path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    # fmt = '%(asctime)s  ' \
    fmt = '[%(levelname)-10s] %(name)-10s : %(message)s'
    # fmt = '[{levelname}] {name} {message}'
    formatter = logging.Formatter(fmt=fmt, style='%')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if path:
        file_handler = logging.FileHandler(filename=path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
