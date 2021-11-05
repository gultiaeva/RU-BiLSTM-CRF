from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer

from common.dataset_reader import UniversalDependenciesDatasetReader


def get_conllu_data_loader(path_to_data,
                           index_with_vocab=None,
                           batch_size=None,
                           shuffle=False,
                           max_instances_in_memory=None,
                           use_elmo_token_indexer=False,
                           cuda_device=None,
                           **kwargs, ):
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
