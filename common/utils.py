import logging

from torch import cuda

logging.getLogger(__name__)


def get_cuda_device_if_available():

    if cuda.is_available():
        cuda_device = 0  # GPU
        device_name = cuda.get_device_name()
        logging.info(f'CUDA device is available. Using {device_name}.')
    else:
        cuda_device = -1  # CPU
        logging.info('No CUDA device detected. Using CPU.')

    return cuda_device
