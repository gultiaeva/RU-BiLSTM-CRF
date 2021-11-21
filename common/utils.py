import logging

from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from torch import cuda

from common.dataset_reader import UniversalDependenciesDatasetReader, SimpleStringReader

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


def get_conllu_data_loader(path_to_data,
                           index_with_vocab=None,
                           batch_size=None,
                           shuffle=False,
                           max_instances_in_memory=None,
                           use_elmo_token_indexer=False,
                           cuda_device=None,
                           **kwargs):
    if use_elmo_token_indexer:
        token_indexer = {'elmo_tokens': ELMoTokenCharactersIndexer()}
    else:
        token_indexer = None

    reader = UniversalDependenciesDatasetReader(token_indexers=token_indexer)
    data_loader = MultiProcessDataLoader(
        reader=reader,
        data_path=path_to_data,
        batch_size=batch_size,
        shuffle=shuffle,
        max_instances_in_memory=max_instances_in_memory,
        cuda_device=cuda_device,
        **kwargs
    )
    if index_with_vocab is not None:
        data_loader.index_with(index_with_vocab)
    return data_loader


def get_string_reader(use_elmo_token_indexer):
    if use_elmo_token_indexer:
        token_indexer = {'elmo_tokens': ELMoTokenCharactersIndexer()}
    else:
        token_indexer = None

    return SimpleStringReader(token_indexers=token_indexer)
