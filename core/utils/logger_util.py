import logging
import sys
import functools
from core.cv_utils import create_if_not_exist
from config import Config


create_if_not_exist(Config.Dir.DATA_DIR)
create_if_not_exist(Config.Dir.LOG_DIR)


class LogService(object):
    logger = None

    def __init__(self, logger):
        self.stdout = sys.stdout
        self.logger = logger
        self.logger.setLevel(logging.NOTSET)
        # self.stderr = sys.stderr
        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s - %(processName)s: %(message)s")

        file_handler = logging.FileHandler(Config.Log.LOG_FILE, mode='a')
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

    def write(self, message):
        self.stdout.write(message)

        if message != '\n':
            self.logger.debug(message)

    def flush(self):
        pass

    @staticmethod
    def exception_handler(logger, type, value, tb):
        logger.exception("Uncaught exception: {0}".format(str(value)))


logger = logging.getLogger("annotation_service")
log_service = LogService(logger)
sys.excepthook = functools.partial(LogService.exception_handler, logger)
sys.stdout = log_service
print('Run logger')
