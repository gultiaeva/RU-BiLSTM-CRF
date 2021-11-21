import logging
import os
from datetime import datetime


LOGS_DIR = 'logs'

logging.getLogger().setLevel(logging.DEBUG)

dt = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
filename = f'logs_{dt}.log'
path_to_log = os.path.join(LOGS_DIR, filename)

file_handler = logging.FileHandler(path_to_log, mode='a')
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)

logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(stream_handler)
