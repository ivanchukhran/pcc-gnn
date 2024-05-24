import os
import sys
import logging


def setup_logging(path: str) -> logging.Logger:
    """
    Setup the logging for the application.

    Arguments:
        path: str - The path to save the log file.

    Returns:
        logging.Logger - The logger to use.
    """
    os.makedirs(path, exist_ok=True)
    logs_path = os.path.join(path, 'logs.log')
    write_mode = 'w' if not os.path.exists(logs_path) else 'a'
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=logs_path,
                        filemode=write_mode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(console)
    return logger
