import os
import logging
import time

logger = None

def create_logger(config, log_save_dir):
    log_file = '{}_{}.log'.format(config.experiment_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s | %(filename)-10s | line %(lineno)-3d: %(message)s'
    logging.basicConfig(filename=os.path.join(log_save_dir, log_file), format=head)
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(head))
    logger.addHandler(console)
    return logger