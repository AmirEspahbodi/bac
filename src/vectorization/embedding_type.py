from enum import Enum

class EmbeddingType(Enum):
    GLOVE = "glove"
    BERT = "bert"
    BERT_CLS = "bert_cls"

    def __str__(self):
        return self.value