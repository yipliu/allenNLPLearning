from typing import Dict, Iterator, List
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

import itertools






@DatasetReader.register("conll_03_reader")
class CoNLL03DatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        # lazy = True : when dataset is large, set it true. AllenNLP will load dataset from disk in batch-size chunks. 
        # lazy = False: when dataset is small, set it false. AllenNLP will store the dataset in memory

        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self,
                         words: List[str],
                         ner_tags: List[str]) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField([Token(w) for w in words], self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokes"] = tokens
        fields["label"] = SequenceLabelField(ner_tags, tokens)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        # line.strip() == '' means judge value is null or not
        is_divider = lambda line: line.strip() == ''

        # Open the dataset path
        with open(file_path, 'r') as conll_file:
            for divider, lines in itertools.groupby(conll_file, is_divider):
                '''
                divider = bool -> line.strip() == ''
                True: this line is equal to Null
                False: this line has some words 
                '''
                #
                if not divider:
                    fields = [l.strip().split() for l in lines]
                    fields = [l for l in zip(*fields)]
                    tokens, _, _, ner_tags = fields
                    yield self.text_to_instance(tokens, ner_tags)
                '''
                l = the result of each line that in txt except NULL

                if l = 'BRUSSELS NNP I-NP I-LOC'
                l.strip().split() = ['BRUSSELS', 'NNP', 'I-NP', 'I-LOC']

                fields = [['BRUSSELS', 'NNP', 'I-NP', 'I-LOC']]
                '''

                    # switch it so that each field is a list of tokens/labels
                    
                '''
                zip(*fields) = make every column in a tuple
                fields = [element]
                e.g.: [(col1),(col2),(col3),(col4)]
                '''

                    # only keep the tokens and NER labels
                    # in fields, col1 = tokens, col4 = label
                
