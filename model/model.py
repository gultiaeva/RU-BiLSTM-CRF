# -*- coding: utf-8 -*-
from typing import Dict, Any

import torch.nn as nn
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp_models.tagging import CrfTagger


@Model.register("bilstm-crf")
class BiLSTMCRF(CrfTagger):
    def __init__(self, vocab,
                 use_elmo=False, elmo_options_file=None, elmo_weights_file=None,
                 embed_dim=172, hidden_dim=256, dropout=.2):

        self.use_elmo = use_elmo
        self.elmo_options_file = elmo_options_file
        self.elmo_weights_file = elmo_weights_file
        self.hidden_dim = hidden_dim

        if self.use_elmo:
            token_embedding = ElmoTokenEmbedder(
                options_file=self.elmo_options_file,
                weight_file=self.elmo_weights_file
            )
            embed_dim = token_embedding.output_dim
            word_embeddings = BasicTextFieldEmbedder({'elmo_tokens': token_embedding})
        else:
            token_embedding = Embedding(
                num_embeddings=vocab.get_vocab_size('tokens'),
                embedding_dim=embed_dim
            )
            word_embeddings = BasicTextFieldEmbedder({'tokens': token_embedding})

        bidirectional_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
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
        )


