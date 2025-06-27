from enum import Enum

class EmbeddingType(Enum):
    GLOVE = "glove"
    W2V = "w2v"
    BERT = "bert"
    ST = "ST"

    def __str__(self):
        return self.value