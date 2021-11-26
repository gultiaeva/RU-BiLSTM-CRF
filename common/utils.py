import logging
import os
import shutil
from typing import Union

import torch.types
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from torch import cuda
from torch import device

from common.dataset_reader import UniversalDependenciesDatasetReader, SimpleStringReader

logging.getLogger(__name__)


def wipe_dir(path_to_dir: str) -> None:
    """
    Deletes all files in specified directory.

    :param path_to_dir: Directory to wipe.
    :type path_to_dir: str

    :return: None
    """
    shutil.rmtree(path_to_dir, ignore_errors=True)
    os.mkdir(path_to_dir)


def create_dir_if_not_exists(path_to_dir: str) -> None:
    """
    Creates directory if it is not exists.

    :param path_to_dir: Path to create directory.
    :type path_to_dir: str

    :return: None
    """
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


def is_empty_dir(path_to_dir: str) -> bool:
    """
    Checks if directory is empty or only contains hidden files (.filename)

    :param path_to_dir: Path to check.
    :type path_to_dir: str

    :return: True if directory is empty else False
    :rtype: bool
    """
    # List files except hidden (starting with ".")
    files = list(filter(lambda x: not x.startswith('.'), os.listdir(path_to_dir)))
    return not bool(files)


def path_exists(path: str) -> bool:
    """
    Checks  specified path exists.

    :param path: Path to check
    :type path: str

    :return: True if path exists else False
    :rtype: bool
    """
    return os.path.exists(path)


def get_cuda_device_if_available() -> torch.types.Device:
    """
    Checks if any CUDA devices is available and returns it.

    :return: 0 if CUDA device is available else 0

    :rtype: int
    """

    if cuda.is_available():
        cuda_device = f'cuda:0'  # GPU
        device_name = cuda.get_device_name()
        logging.info(f'CUDA device is available. Using {device_name}.')
    else:
        cuda_device = "cpu"  # CPU
        logging.info('No CUDA device detected. Using CPU.')

    return device(cuda_device)


def get_conllu_data_loader(path_to_data: str,
                           index_with_vocab: Vocabulary,
                           batch_size: Union[int, None] = None,
                           shuffle: bool = False,
                           max_instances_in_memory: Union[int, None] = None,
                           use_elmo_token_indexer: bool = False,
                           cuda_device: Union[int, None] = None,
                           **kwargs) -> MultiProcessDataLoader:
    """
    Sets up data loader to work with UniversalDependenciesDatasetReader and embeddings if provided.

    :param path_to_data: Path to dataset in CoNLL-U format (gzip compressed).
    :type path_to_data: str
    :param index_with_vocab: Vocabulary to index dataset.
    :type index_with_vocab: allennlp.data.vocabulary.Vocabulary
    :param batch_size: Size (in sentences, not in documents) of single batch. None for non-batch reading.
    :type batch_size: int or None
    :param shuffle: Provide shuffling in batches.
    :type shuffle: bool
    :param max_instances_in_memory: Maximum sentences that can be stored in memory at once. None for no limitation.
    :type max_instances_in_memory: int or None
    :param use_elmo_token_indexer: Index tokens with ELMoTokenCharactersIndexer for ELMo embeddings.
        Use only if you use ELMo embeddings in tour model.
    :type use_elmo_token_indexer: bool
    :param cuda_device: Device (CPU or any available CUDA device) to store data.
    :type cuda_device: int or None
    :param kwargs: Any keyword arguments for allennlp.data.data_loaders.multiprocess_data_loader.MultiProcessDataLoader

    :return: Set up dataset reader
    :rtype: allennlp.data.data_loaders.multiprocess_data_loader.MultiProcessDataLoader
    """
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


def get_string_reader(use_elmo_token_indexer: bool) -> SimpleStringReader:
    """
    Sets up sting reader for predictions from raw text.

    :param use_elmo_token_indexer: Index tokens with ELMoTokenCharactersIndexer for ELMo embeddings.
        Use only if you use ELMo embeddings in tour model.
    :type use_elmo_token_indexer: bool

    :return: Set up string reader
    :rtype: common.dataset_reader.SimpleStringReader
    """
    if use_elmo_token_indexer:
        token_indexer = {'elmo_tokens': ELMoTokenCharactersIndexer()}
    else:
        token_indexer = None

    return SimpleStringReader(token_indexers=token_indexer)
