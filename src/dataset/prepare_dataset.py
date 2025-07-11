from ._1_load_dataset import load_dataset
from ._2_splitting import slit_dataset
from ._3_data_augmentation import contextual_data_augmentation
from ._3_2_data_augmentation import contextual_word_replacement_augmentation
from src.vectorization import data_loaders_with_glove, get_data_loaders_bert, get_data_loaders_bert_cls
from ._dataset_types import DatasetType
from src.vectorization import EmbeddingType
from src.tokenization import get_bert_tokenizer
from src.utils import get_single_vector_dataloader


def get_data_loaders(dataset_type: DatasetType, embedding_type:EmbeddingType, remove_stop_words=False):
    dataset, NUM_ACTUAL_CLS = load_dataset(
        dataset_type
    )
    train_dataset, test_dataset, validation_dataset = slit_dataset(dataset)
    aug_train_dataset = contextual_word_replacement_augmentation(
        train_dataset, dataset_type
    )
    bert_tokenizer = get_bert_tokenizer()
    
    match embedding_type:
        case EmbeddingType.GLOVE:
            aug_train_loader, val_loader, test_loader = data_loaders_with_glove(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
                remove_stop_words=remove_stop_words
            )
        case EmbeddingType.GLOVE_MEAN:
            aug_train_loader, val_loader, test_loader = data_loaders_with_glove(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
                remove_stop_words=remove_stop_words
            )
            aug_train_loader = get_single_vector_dataloader(aug_train_loader)
            val_loader = get_single_vector_dataloader(val_loader)
            test_loader = get_single_vector_dataloader(test_loader)
        case EmbeddingType.BERT:
            aug_train_loader, val_loader, test_loader = get_data_loaders_bert(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
                remove_stop_words=remove_stop_words
            )
        case EmbeddingType.BERT_CLS:
            aug_train_loader, val_loader, test_loader = get_data_loaders_bert_cls(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
                remove_stop_words=remove_stop_words
            )
        case EmbeddingType.BERT_MEAN:
            aug_train_loader, val_loader, test_loader = get_data_loaders_bert(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
                remove_stop_words=remove_stop_words
            )
            aug_train_loader = get_single_vector_dataloader(aug_train_loader)
            val_loader = get_single_vector_dataloader(val_loader)
            test_loader = get_single_vector_dataloader(test_loader)

    return aug_train_loader, val_loader, test_loader, NUM_ACTUAL_CLS
