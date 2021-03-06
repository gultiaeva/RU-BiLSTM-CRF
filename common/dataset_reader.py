import gzip
import logging
from typing import Dict, Tuple, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from conllu import parse_incr
from overrides import overrides

logger = logging.getLogger(__name__)


# Modified UniversalDependenciesDatasetReader from
# allennlp_models.structured_prediction.dataset_readers.universal_dependencies
@DatasetReader.register("universal_dependencies", exist_ok=True)
class UniversalDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : `bool`, optional (default = `False`)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : `Tokenizer`, optional (default = `None`)
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        tokenizer: Tokenizer = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos
        self.tokenizer = tokenizer

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with gzip.open(file_path, "rt", encoding="utf8") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by integers here as elided words have a non-integer word id,
                # as parsed by the conllu python library.
                annotation = [x for x in annotation if isinstance(x["id"], int)]

                heads = []
                tags = []
                words = []
                labels = []
                pos_tags = []

                if self.use_language_specific_pos:
                    pos_tags_namespace = "xpostag"
                else:
                    pos_tags_namespace = "upostag"

                for x in annotation:
                    heads.append(x["head"])
                    tags.append(x["deprel"])
                    words.append(x["form"])
                    labels.append(x["misc"]["Tag"])
                    pos_tags.append(x[pos_tags_namespace])

                yield self.text_to_instance(words, pos_tags, labels, list(zip(tags, heads)))

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        upos_tags: List[str],
        labels: List[str],
        dependencies: List[Tuple[str, int]] = None,
    ) -> Instance:

        """
        # Parameters

        words : `List[str]`, required.
            The words in the sentence to be encoded.
        upos_tags : `List[str]`, required.
            The universal dependencies POS tags for each word.
        dependencies : `List[Tuple[str, int]]`, optional (default = `None`)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        # Returns

        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(words))
        else:
            tokens = [Token(t) for t in words]

        text_field = TextField(tokens, self._token_indexers)
        fields["tokens"] = text_field
        fields["tags"] = SequenceLabelField(labels, text_field, label_namespace="labels")
        fields["pos_tags"] = SequenceLabelField(upos_tags, text_field, label_namespace="pos")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                [x[0] for x in dependencies], text_field, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                [x[1] for x in dependencies], text_field, label_namespace="head_index_tags"
            )

        fields["metadata"] = MetadataField({"words": words, "labels": labels, "pos": upos_tags})
        return Instance(fields)


@DatasetReader.register("simple_string_reader", exist_ok=True)
class SimpleStringReader(DatasetReader):
    """
    Simple wrapper to work with SentenceTaggerPredictor.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    """

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, tokens: List[Token]) -> Instance:

        fields: Dict[str, Field] = {}

        text_field = TextField(tokens, self._token_indexers)
        fields["tokens"] = text_field
        fields["metadata"] = MetadataField({"words": tokens})
        return Instance(fields)
