from enum import Enum

class EmbeddingType(Enum):
    GLOVE = "glove"
    BERT = "bert"

    def __str__(self):
        return self.value