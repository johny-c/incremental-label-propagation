import logging
import sys


class BaseClass:

    obj_count = 0

    def __init__(self, name):

        type(self).obj_count += 1
        self.name = name + '(' + str(type(self).obj_count) + ')'
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # fmt = '%(asctime)s  ' \
        fmt = '[%(levelname)-6s] %(name)-8s : %(message)s'
        # fmt = '[{levelname}] {name} {message}'
        formatter = logging.Formatter(fmt=fmt, style='%')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.logger.info('Created instance of {}.'.format(type(self).__name__))
