from ._1_load_dataset import load_dataset
from ._2_splitting import slit_dataset
from ._3_data_augmentation import contextual_data_augmentation
from ._3_2_data_augmentation import contextual_word_replacement_augmentation
from src.vectorization import data_loaders_with_glove, get_data_loaders_bert
from ._dataset_types import DatasetType
from src.vectorization import EmbeddingType
from src.tokenization import get_bert_tokenizer


def get_data_loaders(embedding_type:EmbeddingType):
    dataset, NUM_ACTUAL_CLS = load_dataset(
        DatasetType.GCC, dataset_rpath="./datasets/gcc_data.csv"
    )
    train_dataset, test_dataset, validation_dataset = slit_dataset(dataset)
    aug_train_dataset = contextual_word_replacement_augmentation(
        train_dataset, DatasetType.GCC
    )
    bert_tokenizer = get_bert_tokenizer()
    
    match embedding_type:
        case EmbeddingType.GLOVE:
            print("here start")
            aug_train_loader, val_loader, test_loader = data_loaders_with_glove(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
            )
            print("here end")
        case EmbeddingType.BERT:
            aug_train_loader, val_loader, test_loader = get_data_loaders_bert(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
            )
    return aug_train_loader, val_loader, test_loader, NUM_ACTUAL_CLS
