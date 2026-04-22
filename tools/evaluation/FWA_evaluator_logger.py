from logging import getLogger, handlers, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL
from config import Config

def set_logger():

    root_logger = getLogger()
    root_logger.setLevel(Config.LOG_LEVEL)
    rotating_handler = handlers.RotatingFileHandler(
        filename='FWA_evaluator.log',
        mode='a',
        maxBytes=1000000,
        backupCount=3,
        encoding='utf-8'
    )

    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rotating_handler.setFormatter(formatter)

    root_logger.addHandler(rotating_handler)
