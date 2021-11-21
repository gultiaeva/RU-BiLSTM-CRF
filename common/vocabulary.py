import logging

from allennlp.data.vocabulary import Vocabulary

from common.dataset_reader import UniversalDependenciesDatasetReader
from common.utils import wipe_dir
logging.getLogger(__name__)


def build_vocab(*paths, save_dir_path, overwrite=True):
    if overwrite:
        wipe_dir(save_dir_path)
    reader = UniversalDependenciesDatasetReader()
    vocabulary = Vocabulary()
    for path in paths:
        dataset = reader.read(path)
        vocabulary.extend_from_instances(dataset)

    vocabulary.save_to_files(save_dir_path)


def load_vocab(path_to_vocab_dir):
    vocabulary = Vocabulary.from_files(path_to_vocab_dir)
    return vocabulary


if __name__ == '__main__':
    train_path = 'data/dataset/train_data.conllu.gz'
    test_path = 'data/dataset/test_data.conllu.gz'
    validation_path = 'data/dataset/validation_data.conllu.gz'
    datasets_paths = [train_path, test_path, validation_path]

    vocab_dir = 'data/vocab'

    build_vocab(*datasets_paths, save_dir_path=vocab_dir)

    vocab = load_vocab(vocab_dir)
