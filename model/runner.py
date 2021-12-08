import logging
from typing import Union, Dict, Any

import spacy
import torch
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from torch import device
from torch.optim import Adam, Optimizer

from common import MetricsLoggerCallback
from common.utils import replace_string
from common.utils import get_conllu_data_loader, get_string_reader
from common.utils import get_cuda_device_if_available
from common.utils import path_exists, create_dir_if_not_exists, is_empty_dir
from common.vocabulary import load_vocab
from model import BiLSTMCRF

logging.getLogger(__name__)


class NERModel:
    """
    Model for Named Entity Recognition.

    The `NERModel` class wraps building BiLSTM+CRF/BiGRU+CRF model, reading dataset in batches, selecting an optimizer
    setting up early stopping and serialization directories, train and predict processes.

    :param model_name: Model name for further identification purpose.
    :type model_name: str
    :param vocabulary_dir: Path to directory with dataset vocabulary.
    :type vocabulary_dir: str
    :param train_dataset_file: Path to training dataset file.
    :type train_dataset_file: str
    :param test_dataset_file: Path to testing dataset file.
    :type test_dataset_file: str
    :param use_elmo_embeddings: Use pretrained ELMo model to embed sequence.
    :type use_elmo_embeddings: bool
    :param elmo_options_file: Path to pretrained ELMo model options file (Usually named options.json).
        Used only if `use_elmo` is `True`.
    :type elmo_options_file: str or None
    :param elmo_weights_file:  Path to pretrained ELMo model weights file (usually named model.hdf5).
        Used only if `use_elmo` is `True`.
    :type elmo_weights_file: str or None
    :param use_gru_instead_of_lstm: Set up GRU instead of LSTM.
    :type use_gru_instead_of_lstm: bool
    :param embedding_dim: Embedding dimension. Used only if `use_elmo` is `False`.
        If `use_elmo` is `True` then uses embedding dimension from pretrained ELMo model.
    :type embedding_dim: int
    :param hidden_dim: Hidden dimension in Seq2Seq model.
    :type hidden_dim: int
    :param dropout: Dropout regularization. Disables random neuron with `dropout` probability on training iterations.
    :type dropout: float
    :param learning_rate: Determines the step size of optimization algorithm
    :type learning_rate: float
    :param optimizer: Model optimization algorithm for training.
    :type optimizer: torch.optim.Optimizer or None
    :param checkpoints_dir: Directory to store model checkpoints. If `None` then no checkpoints will be created.
        If provided then checkpoints will be created every hour of training.
    :type checkpoints_dir: str or None
    :param model_serialization_dir: Model serialization directory.
        If provided then model weights will be serialized in this directory after training is completed.
        If None then model will not be serialized after training process.
    :type model_serialization_dir: str or None
    :param use_cuda: Usage of CUDA device.
        If `True` then all operations will be performed on CUDA device if it available.
        If `False` then all operations will be performed on CPU, regardless of availability of CUDA.
    :type use_cuda: bool
    """
    def __init__(self,
                 model_name: str,
                 vocabulary_dir: str,
                 train_dataset_file: str,
                 test_dataset_file: str,
                 use_elmo_embeddings: bool = False,
                 elmo_options_file: Union[str, None] = None,
                 elmo_weights_file: Union[str, None] = None,
                 use_gru_instead_of_lstm: bool = False,
                 embedding_dim: int = 172,
                 hidden_dim: int = 256,
                 dropout: float = .1,
                 learning_rate: float = 0.01,
                 optimizer: Union[Optimizer, None] = None,
                 checkpoints_dir: Union[str, None] = None,
                 model_serialization_dir: Union[str, None] = None,
                 use_cuda: bool = True) -> None:
        """
        :raises: FileNotFoundError if no vocabulary detected.
        """

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
        else:
            self.device = device('cpu')

        self.model.to(self.device)

        # Optimizer init
        params = self.model.parameters()
        self.optimizer = optimizer or Adam(params, lr=learning_rate)

        # Model checkpoints
        if checkpoints_dir:
            if not path_exists(checkpoints_dir):
                create_dir_if_not_exists(checkpoints_dir)
            self.checkpoints = Checkpointer(checkpoints_dir, save_every_num_seconds=3600, keep_most_recent_by_count=25)
        else:
            self.checkpoints = None

        self._is_predictor_initialized = False
        self._is_model_trained = False
        self._spacy_tokenizer_name = 'ru_core_news_sm'

    def get_info(self) -> Dict[str, Any]:
        """
        Gets essential model params.

        :return: Model params
        :rtype: dict
        """
        info = {
            'model_name': self.model_name,
            'has_elmo_embeddings': self.use_elmo_embeddings,
            'GRU': self.model.use_gru_instead_of_lstm,
            'embedding_dim': self.model.embedding_dim,
            'hidden_dim': self.model.hidden_dim,
            'dropout': self.model.dropout.p
        }
        return info

    def load_model_state(self, checkpoint_path: str) -> None:
        """
        Loads model state from file (with extension `.th`).

        :param checkpoint_path:
        :return:
        """
        with open(checkpoint_path, 'rb') as model_state:
            state_dict = torch.load(model_state, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self._is_model_trained = True

    def fit(self,
            epochs: int = 20,
            early_stopping_patience: int = 3,
            batch_size: int = 256,
            shuffle: bool = False,
            max_instances_in_memory: int = 1000) -> None:
        """
        Launches train process.

        :param epochs: Number of epochs of training.
        :type epochs: int
        :param early_stopping_patience: Number of epochs to be patient before early stopping:
            the training is stopped after patience epochs with no improvement.
            If given, it must be > 0. If None, early stopping is disabled.
        :type early_stopping_patience: int
        :param batch_size: Size of a single training batch (in sentences, not in documents).
        :type batch_size: int
        :param shuffle: Provide shuffling in batches.
        :type shuffle: bool
        :param max_instances_in_memory: Maximum sentences that can be stored in memory at once. None for no limitation.
        :type max_instances_in_memory: int

        :return: None
        """

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

        if self.device != device('cpu'):
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

    def _init_predictor(self) -> None:
        """
        Initializes SentenceTaggerPredictor if not initialized yet.

        :return: None
        """
        reader = get_string_reader(use_elmo_token_indexer=self.use_elmo_embeddings)
        self._tokenizer = spacy.load(self._spacy_tokenizer_name)
        self._predictor = SentenceTaggerPredictor(self.model, reader, language=self._spacy_tokenizer_name)
        self._predictor_initialized = True

    def anonymize_sentence(self, sentence: str) -> str:
        """
        Replaces all Named Entities with their types.
        Example:
            >>> input_string = 'Иван Васильевич меняет профессию'
            >>> result_string = model.anonymize_sentence(input_string)
            >>> print(result_string)
            '[PER] меняет профессию'

        :param sentence: String that need to be anonymized
        :type sentence: str

        :return: String with deleted Named Entities
        :rtype: str
        """
        # Check if model is fitted and predictor initialized (for first method run)
        assert self._is_model_trained, 'Model is not trained! You must fit model first.'
        if not self._is_predictor_initialized:
            self._init_predictor()

        # Get token indices and lengths from original string using the same spacy tokenizer as .predict
        tokens_info = {token.i: (token.idx, len(token)) for token in self._tokenizer(sentence)}
        # Get predicted token tags
        prediction = self.predict(sentence)
        tags = prediction['tags']

        tags_to_replace = []  # List of (tag, [start_idx, end_idx]). Start and end indices are from original string
        prev_tag_grp = 'O'
        for i, tag in enumerate(tags):
            # Skip if tagged as O
            if tag == 'O':
                prev_tag_grp = 'O'
                continue
            # Get rid of "B-" and "I-" in tags names.
            tag_grp = tag[-3:]
            # Merge complex B->I->...->I sequences into one tag. e.g. B-LOC -> I-LOC -> I-LOC will be merged as one LOC
            if tag_grp != prev_tag_grp:
                # If new tag encountered then get its start and end positions
                from_idx, length = tokens_info[i]
                append_obj = tag_grp, [from_idx, from_idx+length]
                tags_to_replace.append(append_obj)
                prev_tag_grp = tag_grp
            else:
                # If tag is a part of a sequence then replace last tag end index with end index of a current tag
                from_idx, length = tokens_info[i]
                last_tag = tags_to_replace[-1]
                last_tag_indices = last_tag[-1]
                last_tag_indices[-1] = from_idx + length

        # Replace original string with tags
        sent = sentence
        for named_entity_type, (from_idx, to_idx) in reversed(tags_to_replace):
            named_entity_type_repl = f'[{named_entity_type}]'
            sent = replace_string(sent, from_idx, to_idx, named_entity_type_repl)

        return sent

    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Splits raw text into tokens and then predicts label to each token.

        :param sentence: Raw text for prediction.
        :type sentence: str

        :return: Tagged sentence with some meta information.
        :rtype: dict
        """
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

    checkpoints_directory = 'data/models/model_lstm_elmo/checkpoints'
    serialization_directory = 'data/models/model_lstm_elmo'

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
