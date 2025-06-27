from ._1_load_dataset import load_dataset
from ._2_splitting import slit_dataset
from ._3_data_augmentation import contextual_data_augmentation
from ._3_2_data_augmentation import contextual_word_replacement_augmentation
from ._4_data_loaders import create_data_loaders_word_embedding, create_data_loaders_contectualized
from ._dataset_types import DatasetType
from src.vectorization import EmbeddingType, load_glove, get_word2vec_vectors
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
            aug_train_loader, val_loader, test_loader = create_data_loaders_word_embedding(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
                vectorization_function=load_glove,
            )
        case EmbeddingType.W2V:
            aug_train_loader, val_loader, test_loader = create_data_loaders_word_embedding(
                aug_train_dataset,
                test_dataset,
                validation_dataset,
                bert_tokenizer,
                vectorization_function=get_word2vec_vectors,
            )
        case EmbeddingType.BERT:
            aug_train_loader, val_loader, test_loader = create_data_loaders_contectualized(
                train_df=aug_train_dataset,
                validation_df=test_dataset,
                test_df=validation_dataset,
                bert_tokenizer=bert_tokenizer,
                batch_size=3, # Small batch size for demonstration
                embedding_type=EmbeddingType.BERT
            )
        case EmbeddingType.ST:
            aug_train_loader, val_loader, test_loader = create_data_loaders_contectualized(
                train_df=aug_train_dataset,
                validation_df=test_dataset,
                test_df=validation_dataset,
                bert_tokenizer=bert_tokenizer,
                batch_size=3, # Small batch size for demonstration
                embedding_type=EmbeddingType.ST
            )
    return aug_train_loader, val_loader, test_loader, NUM_ACTUAL_CLS
