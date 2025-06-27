from ._1_load_dataset import load_dataset
from ._2_splitting import slit_dataset
from ._3_data_augmentation import contextual_data_augmentation
from ._3_2_data_augmentation import contextual_word_replacement_augmentation
from ._4_data_loaders import create_data_loaders_word_embedding#, create_data_loaders_contectualized
from ._dataset_types import DatasetType
from src.vectorization import VectorizationsType
from src.tokenization import get_bert_tokenizer


def get_data_loaders(vectiriation_function, vectorizations_type:VectorizationsType):
    dataset, NUM_ACTUAL_CLS = load_dataset(
        DatasetType.GCC, dataset_rpath="./datasets/gcc_data.csv"
    )
    train_dataset, test_dataset, validation_dataset = slit_dataset(dataset)
    aug_train_dataset = contextual_word_replacement_augmentation(
        train_dataset, DatasetType.GCC
    )
    bert_tokenizer = get_bert_tokenizer()
    
    match vectorizations_type:
        case VectorizationsType.WORD_EMBEDDING:
            aug_train_loader, val_loader, test_loader = create_data_loaders_word_embedding(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
                vectorization_function=vectiriation_function,
            )
        case VectorizationsType.CONTECTUALIZED_EMBEDDINGS:
            aug_train_loader, val_loader, test_loader = create_data_loaders_contectualized(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
                vectorization_function=vectiriation_function,
            )
        
    return aug_train_loader, val_loader, test_loader, NUM_ACTUAL_CLS
