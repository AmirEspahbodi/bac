from ._1_load_dataset import load_dataset
from ._2_splitting import slit_dataset
from ._3_data_augmentation import create_contextual_augmentation
from ._4_data_loaders import create_data_loaders
from ._dataset_types import DatasetType
from src.vectorization import load_glove
from src.tokenization import get_bert_tokenizer


def get_data_loaders():
    dataset, NUM_ACTUAL_CLS = load_dataset(
        DatasetType.GCC, dataset_rpath="./datasets/gcc_data.csv"
    )
    train_dataset, test_dataset, validation_dataset = slit_dataset(dataset)
    aug_train_dataset = create_contextual_augmentation(train_dataset)
    bert_tokenizer = get_bert_tokenizer()
    aug_train_loader, val_loader, test_loader = create_data_loaders(
        aug_train_dataset,
        test_dataset,
        validation_dataset,
        bert_tokenizer,
        vectorization_function=load_glove,
    )
    return aug_train_loader, val_loader, test_loader, NUM_ACTUAL_CLS
