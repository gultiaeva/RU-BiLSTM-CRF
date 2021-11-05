import logging
import sys

from allennlp.data.vocabulary import Vocabulary

from common.dataset_reader import UniversalDependenciesDatasetReader

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def build_vocab(*paths, save_dir_path):
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
    train_path = '../data/dataset/unzipped/train_data.conllu'
    test_path = '../data/dataset/unzipped/test_data.conllu'
    validation_path = '../data/dataset/unzipped/validation_data.conllu'
    vocab_dir = '../data/vocab'

    build_vocab(train_path, test_path, validation_path, save_dir_path=vocab_dir)

    vocab = load_vocab(vocab_dir)
