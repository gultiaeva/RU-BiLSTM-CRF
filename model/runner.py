import logging

import torch
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from torch.optim import Adam

from common import MetricsLoggerCallback
from common.utils import get_conllu_data_loader, get_string_reader
from common.utils import get_cuda_device_if_available
from common.utils import path_exists, create_dir_if_not_exists, is_empty_dir
from common.vocabulary import load_vocab
from model import BiLSTMCRF

logging.getLogger(__name__)


class NERModel:
    def __init__(self, model_name, vocabulary_dir, train_dataset_file, test_dataset_file,
                 use_elmo_embeddings=False, elmo_options_file=None, elmo_weights_file=None,
                 use_gru_instead_of_lstm=False, embedding_dim=172, hidden_dim=256, dropout=.1, learning_rate=0.01,
                 optimizer=None, checkpoints_dir=None, model_serialization_dir=None, use_cuda=True):

        self.model_name = model_name

        # Datasets
        self.train_dataset_file = train_dataset_file
        self.test_dataset_file = test_dataset_file

        # Pretrained ELMo embeddings
        self.use_elmo_embeddings = use_elmo_embeddings
        self.elmo_options_file = elmo_options_file
        self.elmo_weights_file = elmo_weights_file

        # Serialization
        self.model_serialization_directory = model_serialization_dir
        if self.model_serialization_directory and not path_exists(model_serialization_dir):
            logging.info(f'Directory {model_serialization_dir} is not exists. Creating it...')
        create_dir_if_not_exists(model_serialization_dir)

        # Vocabulary
        if not path_exists(vocabulary_dir) or is_empty_dir(vocabulary_dir):
            FileNotFoundError(f'No vocabulary detected at {vocabulary_dir}. You have to build vocabulary first!')

        self.vocabulary = load_vocab(vocabulary_dir)

        # Model
        self.model = BiLSTMCRF(
            self.vocabulary,
            use_elmo=use_elmo_embeddings,
            elmo_options_file=elmo_options_file,
            elmo_weights_file=elmo_weights_file,
            use_gru_instead_of_lstm=use_gru_instead_of_lstm,
            embed_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        model_description = self.get_info()
        logging.info(','.join(f'{k}={v}' for k, v in model_description.items()))

        # CUDA settings
        if use_cuda:
            self.device = get_cuda_device_if_available()
            self.model.cuda(self.device)
        else:
            self.device = -1

        # Optimizer init
        params = self.model.parameters()
        self.optimizer = optimizer or Adam(params, lr=learning_rate)

        # Model checkpoints
        if checkpoints_dir:
            if not path_exists(checkpoints_dir):
                create_dir_if_not_exists(checkpoints_dir)
            self.checkpoints = Checkpointer(checkpoints_dir, save_every_num_seconds=3600)
        else:
            self.checkpoints = None

        self._is_predictor_initialized = False
        self._is_model_trained = False

    def get_info(self):
        info = {
            'model_name': self.model_name,
            'has_elmo_embeddings': self.use_elmo_embeddings,
            'GRU': self.model.use_gru_instead_of_lstm,
            'embedding_dim': self.model.embedding_dim,
            'hidden_dim': self.model.hidden_dim,
            'dropout': self.model.dropout.p
        }
        return info

    def load_model_state(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as model_state:
            state_dict = torch.load(model_state)
            self.model.load_state_dict(state_dict)
        self._is_model_trained = True

    def fit(self, epochs=20, early_stopping_patience=3, batch_size=256, shuffle=False, max_instances_in_memory=1000):

        data_loader_train = get_conllu_data_loader(
            path_to_data=self.train_dataset_file,
            index_with_vocab=self.vocabulary,
            shuffle=shuffle,
            batch_size=batch_size,
            max_instances_in_memory=max_instances_in_memory,
            use_elmo_token_indexer=self.use_elmo_embeddings
        )

        data_loader_test = get_conllu_data_loader(
            path_to_data=self.test_dataset_file,
            index_with_vocab=self.vocabulary,
            batch_size=batch_size,
            max_instances_in_memory=max_instances_in_memory,
            use_elmo_token_indexer=self.use_elmo_embeddings
        )

        if self.device != -1:
            data_loader_train.set_target_device(self.device)
            data_loader_test.set_target_device(self.device)

        callback = MetricsLoggerCallback(
            self.model_serialization_directory,
            summary_interval=10,
            should_log_parameter_statistics=False
        )

        trainer = GradientDescentTrainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=data_loader_train,
            validation_data_loader=data_loader_test,
            patience=early_stopping_patience,
            num_epochs=epochs,
            callbacks=[callback],
            serialization_dir=self.model_serialization_directory,
            checkpointer=self.checkpoints,
            cuda_device=self.device
        )

        trainer.train()

        self._is_model_trained = True

    def _init_predictor(self):
        reader = get_string_reader(use_elmo_token_indexer=self.use_elmo_embeddings)
        self._predictor = SentenceTaggerPredictor(self.model, reader, language='ru_core_news_sm')
        self._predictor_initialized = True

    def predict(self, sentence):
        assert self._is_model_trained, 'Model is not trained! You must fit model first.'
        if not self._is_predictor_initialized:
            self._init_predictor()

        return self._predictor.predict(sentence)


if __name__ == '__main__':
    vocab_dir = 'data/vocab'
    train_file = 'data/dataset/train_data.conllu.gz'
    test_file = 'data/dataset/test_data.conllu.gz'

    elmo_options = 'data/embeddings/elmo/options.json'
    elmo_weights = 'data/embeddings/elmo/model.hdf5'

    checkpoints_directory = 'data/models/model_lstm_elmo1/checkpoints'
    serialization_directory = 'data/models/model_lstm_elmo1'

    name = 'BiLSTM+CRF+ELMo'
    elmo_embeddings = True
    gru = False
    cuda = True
    batch = 32
    lr = .005

    model = NERModel(
        name,
        vocab_dir,
        train_file,
        test_file,
        use_gru_instead_of_lstm=gru,
        use_elmo_embeddings=elmo_embeddings,
        elmo_options_file=elmo_options,
        elmo_weights_file=elmo_weights,
        checkpoints_dir=checkpoints_directory,
        model_serialization_dir=serialization_directory,
        learning_rate=lr,
        use_cuda=cuda,
    )

    model.fit(batch_size=batch, epochs=2)

    model_checkpoint = 'data/models/model_lstm_elmo/best.th'
    model.load_model_state(model_checkpoint)

    res = model.predict('Привет, мир!')
    print(res)
