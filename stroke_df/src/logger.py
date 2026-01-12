# הקובץ: src/utils.py
import logging

def setup_logger(name_of_log):
    logger = logging.getLogger(name_of_log)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler('project_log.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger