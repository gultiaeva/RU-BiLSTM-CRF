import logging

from common import configuration
from common.utils import path_exists, is_empty_dir
from common.vocabulary import build_vocab
from model import NERModel

logging.getLogger(__name__)


def main():
    if not path_exists(configuration.vocabulary) or is_empty_dir(configuration.vocabulary):
        logging.info(f'No vocabulary detected at {configuration.vocabulary}. Building vocabulary...')
        build_vocab(configuration.train_data, configuration.test_data, configuration.validation_data,
                    save_dir_path=configuration.vocabulary)

    model = NERModel(
        model_name=configuration.name,
        vocabulary_dir=configuration.vocabulary,
        train_dataset_file=configuration.train_data,
        test_dataset_file=configuration.test_data,
        use_elmo_embeddings=configuration.use_elmo,
        elmo_options_file=configuration.elmo_options,
        elmo_weights_file=configuration.elmo_weights,
        use_gru_instead_of_lstm=configuration.use_gru,
        embedding_dim=configuration.embed_dim,
        hidden_dim=configuration.hidden_dim,
        dropout=configuration.dropout,
        learning_rate=configuration.learning_rate,
        optimizer=None,  # Provide if you want to use custom optimizer
        checkpoints_dir=configuration.checkpoints_dir,
        model_serialization_dir=configuration.serialization_dir,
        use_cuda=configuration.use_cuda
    )
    model.fit(
        epochs=configuration.n_epochs,
        early_stopping_patience=configuration.early_stopping_epochs,
        batch_size=configuration.batch_size,
        shuffle=configuration.batch_shuffle,
        max_instances_in_memory=configuration.max_instances_in_memory
    )


if __name__ == '__main__':
    main()
