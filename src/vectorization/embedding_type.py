from enum import Enum

class EmbeddingType(Enum):
    GLOVE = "glove"
    BERT = "bert"
    BERT_CLS = "bert_cls"
    BERT_MEAN = "bert_mean"
    GLOVE_MEAN = "glove_mean"
    
    def __str__(self):
        return self.value
