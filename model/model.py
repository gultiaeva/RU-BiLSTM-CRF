from typing import Union

import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp_models.tagging import CrfTagger


@Model.register("bilstm-crf")
class BiLSTMCRF(CrfTagger):
    """
    Bidirectional LSTM with CRF (Conditional Random Field) layer.
    The `BiLSTMCRF` embeds a sequence, then uses Seq2Seq model (for ex. `LSTM` or `GRU`) on a sequence,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

    Registered as a `Model` with name "bilstm-crf".

    :param vocab: Vocabulary of dataset.
    :type vocab: allennlp.data.vocabulary.Vocabulary
    :param use_elmo: Use pretrained ELMo model to embed sequence.
    :type use_elmo: bool
    :param elmo_options_file: Path to pretrained ELMo model options file (Usually named options.json).
        Used only if `use_elmo` is `True`.
    :type elmo_options_file: str or None
    :param elmo_weights_file: Path to pretrained ELMo model weights file (usually named model.hdf5).
        Used only if `use_elmo` is `True`.
    :type elmo_weights_file: str or None
    :param use_gru_instead_of_lstm: Set up GRU instead of LSTM.
    :type use_gru_instead_of_lstm: bool
    :param embed_dim: Embedding dimension. Used only if `use_elmo` is `False`.
        If `use_elmo` is `True` then uses embedding dimension from pretrained ELMo model.
    :type embed_dim: int
    :param hidden_dim: Hidden dimension in Seq2Seq model.
    :type hidden_dim: int
    :param dropout: Dropout regularization. Disables random neuron with `dropout` probability on training iterations.
    :type dropout: float
    """
    def __init__(self, vocab: Vocabulary,
                 use_elmo: bool = False,
                 elmo_options_file: Union[str, None] = None,
                 elmo_weights_file: Union[str, None] = None,
                 use_gru_instead_of_lstm: bool = False,
                 embed_dim: int = 172,
                 hidden_dim: int = 256,
                 dropout: float = .1) -> None:

        assert (not use_elmo) or (use_elmo and elmo_options_file and elmo_weights_file), 'No pretrained ELMo provided!'

        self.use_gru_instead_of_lstm = use_gru_instead_of_lstm
        self.use_elmo = use_elmo
        self.elmo_options_file = elmo_options_file
        self.elmo_weights_file = elmo_weights_file
        self.hidden_dim = hidden_dim
        self.embedding_dim = embed_dim

        if self.use_elmo:
            token_embedding = ElmoTokenEmbedder(
                options_file=self.elmo_options_file,
                weight_file=self.elmo_weights_file
            )
            self.embedding_dim = token_embedding.output_dim
            word_embeddings = BasicTextFieldEmbedder({'elmo_tokens': token_embedding})
        else:
            token_embedding = Embedding(
                num_embeddings=vocab.get_vocab_size('tokens'),
                embedding_dim=self.embedding_dim
            )
            word_embeddings = BasicTextFieldEmbedder({'tokens': token_embedding})

        recurrent_layer = nn.GRU if self.use_gru_instead_of_lstm else nn.LSTM
        bidirectional_lstm = recurrent_layer(
            self.embedding_dim,
            self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        encoder = PytorchSeq2SeqWrapper(bidirectional_lstm)
        super().__init__(
            vocab=vocab,
            label_namespace='labels',
            text_field_embedder=word_embeddings,
            encoder=encoder,
            dropout=dropout,
            label_encoding='BIO',
            calculate_span_f1=True,
            verbose_metrics=True
        )
