import os
import sys
import logging


logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logg_dir = "logs"

log_filepath = os.path.join(logg_dir, "running_log.log")

os.makedirs(logg_dir, exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_str,
    handlers = [
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]    
)

logger = logging.getLogger("personal_loan_logger")