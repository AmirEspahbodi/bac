import os
import requests
import zipfile
import logging
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from typing import List, Tuple
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from src.utils import clean_str

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- GloVe Configuration ---
GLOVE_DIR = ".cache/glove"
GLOVE_FILE_NAME = "glove.6B.300d.txt"
GLOVE_EMBEDDING_DIM = 300
GLOVE_PATH = f"./{GLOVE_DIR}/{GLOVE_FILE_NAME}"
GLOVE_ZIP_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_LOCAL_ZIP_PATH = f"{GLOVE_DIR}/glove.6B.zip"
Path(f"{os.getcwd()}/{GLOVE_DIR}").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- GloVe File Handling ---
def ensure_glove_zip_or_txt_file_is_present():
    print("ensure_glove_zip_or_txt_file_is_present 1, 1, 1, 1, 1, 1")
    if os.path.exists(GLOVE_PATH):
        logging.info(f"Found existing GloVe text file: {GLOVE_PATH}")
        return True
    print("ensure_glove_zip_or_txt_file_is_present 2, 2, 2, 2, 2, 2")
    if not os.path.exists(GLOVE_LOCAL_ZIP_PATH):
        print("ensure_glove_zip_or_txt_file_is_present 3, 3, 3, 3, 3, 3")
        logging.info(
            f"GloVe zip file {GLOVE_LOCAL_ZIP_PATH} not found. Attempting to download from {GLOVE_ZIP_URL}..."
        )
        try:
            response = requests.get(GLOVE_ZIP_URL, stream=True, verify=False)
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024
            with (
                open(GLOVE_LOCAL_ZIP_PATH, "wb") as file,
                tqdm(
                    desc=f"Downloading {os.path.basename(GLOVE_ZIP_URL)}",
                    total=total_size_in_bytes,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)
            logging.info(f"Successfully downloaded {GLOVE_LOCAL_ZIP_PATH}")
        except requests.exceptions.RequestException as e:
            print("ensure_glove_zip_or_txt_file_is_present 4, 4, 4, 4, 4, 4")
            logging.error(f"Error downloading GloVe zip file: {e}")
            if os.path.exists(GLOVE_LOCAL_ZIP_PATH):
                os.remove(GLOVE_LOCAL_ZIP_PATH)
            return False
        except Exception as e:
            print("ensure_glove_zip_or_txt_file_is_present 5, 5, 5, 5, 5, 5")
            logging.error(f"An unexpected error occurred during download: {e}")
            if os.path.exists(GLOVE_LOCAL_ZIP_PATH):
                os.remove(GLOVE_LOCAL_ZIP_PATH)
            return False
    else:
        logging.info(f"Found existing GloVe zip file: {GLOVE_LOCAL_ZIP_PATH}")

    logging.info(
        f"Attempting to extract {GLOVE_FILE_NAME} from {GLOVE_LOCAL_ZIP_PATH}..."
    )
    try:
        print(f"ensure_glove_zip_or_txt_file_is_present 6, 6, 6, 6, 6, 6. {os.getcwd()}/{GLOVE_LOCAL_ZIP_PATH}")
        with zipfile.ZipFile(GLOVE_LOCAL_ZIP_PATH, "r") as zip_ref:
            print(f"ensure_glove_zip_or_txt_file_is_present 7, 7, 7, 7, 7, 7. {os.getcwd()}/{GLOVE_LOCAL_ZIP_PATH}")
            print(zip_ref.namelist())
            if GLOVE_FILE_NAME in zip_ref.namelist():
                zip_ref.extract(
                    GLOVE_FILE_NAME, path=os.path.dirname(GLOVE_PATH) or "."
                )
                logging.info(
                    f"Successfully extracted {GLOVE_FILE_NAME} to {GLOVE_PATH}"
                )
                return True
            else:
                print(f"ensure_glove_zip_or_txt_file_is_present 7, 7, 7, 7, 7, 7. Error: {GLOVE_FILE_NAME} not found inside {GLOVE_LOCAL_ZIP_PATH}.")
                logging.error(
                    f"Error: {GLOVE_FILE_NAME} not found inside {GLOVE_LOCAL_ZIP_PATH}."
                )
                logging.info(f"Available files: {zip_ref.namelist()}")
                return False
    except zipfile.BadZipFile as e:
        print(f"ensure_glove_zip_or_txt_file_is_present 8, 8, 8, 8, 8, 8. {GLOVE_LOCAL_ZIP_PATH} {e}")
        logging.error(
            f"Error: {GLOVE_DIR}/{GLOVE_FILE_NAME} is a bad zip file. Please delete it and try again."
        )
        return False
    except Exception as e:
        print(f"ensure_glove_zip_or_txt_file_is_present 8, 8, 8, 8, 8, 8. {e}")
        logging.error(f"An error occurred during extraction: {e}")
        return False
    print("ensure_glove_zip_or_txt_file_is_present 9, 9, 9, 9, 9, 9")


def load_glove_vectors(glove_path, embedding_dim):
    logging.info(
        f"Loading GloVe vectors from {glove_path} with dimension {embedding_dim}..."
    )
    if not os.path.exists(glove_path):
        logging.info(f"Error: GloVe file not found at {glove_path}.")
        return None
    word_to_vec = {}
    try:
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading GloVe"):
                values = line.split()
                word = values[0]
                try:
                    vector = np.asarray(values[1:], dtype="float32")
                    if len(vector) == embedding_dim:
                        word_to_vec[word] = vector
                except ValueError:
                    pass
    except Exception as e:
        logging.error(f"An error occurred while reading the GloVe file: {e}")
        return None
    if not word_to_vec:
        logging.info(f"No word vectors loaded from {glove_path}.")
        return None
    logging.info(f"Successfully loaded {len(word_to_vec)} word vectors.")
    return word_to_vec


def load_glove():
    print("load_glove 1, 1, 1, 1, 1, 1, 1")
    if not ensure_glove_zip_or_txt_file_is_present():
        logging.error("Could not obtain GloVe file. Exiting.")
        exit()

    print("load_glove 2, 2, 2, 2, 2, 2, 2")

    glove_vectors_map = load_glove_vectors(GLOVE_PATH, GLOVE_EMBEDDING_DIM)

    unk_embedding = np.random.rand(GLOVE_EMBEDDING_DIM).astype(
        "float32"
    )  # Random UNK if not in GloVe
    if glove_vectors_map:
        if "[unk]" in glove_vectors_map:
            unk_embedding = glove_vectors_map["[unk]"]
        elif "unk" in glove_vectors_map:
            unk_embedding = glove_vectors_map["unk"]
    else:
        logging.warning(
            "GloVe vectors map is empty or not loaded. Using random UNK embedding. Training will be affected."
        )
        # exit() # Critical error, might be best to exit
    return glove_vectors_map, unk_embedding

## get data loaders first method

# def collate_with_glove(batch, hf_tokenizer, glove_word_vectors, embedding_dimension, unk_word_embedding, max_seq_len):
#     labels_list_from_batch = [item['label'] for item in batch]
#     texts_list_from_batch = [item['text'] for item in batch]
#     labels_tensor = torch.LongTensor(labels_list_from_batch)
#     all_sequences_as_vecs = []
#     for text_item in texts_list_from_batch:
#         string_tokens = hf_tokenizer.tokenize(str(text_item))[:max_seq_len] 
#         if not string_tokens:
#             all_sequences_as_vecs.append(torch.tensor(unk_word_embedding, dtype=torch.float).unsqueeze(0))
#             continue
#         current_sequence_embeddings = [torch.tensor(glove_word_vectors.get(token_str, unk_word_embedding), dtype=torch.float) for token_str in string_tokens]
#         if not current_sequence_embeddings:
#             all_sequences_as_vecs.append(torch.tensor(unk_word_embedding, dtype=torch.float).unsqueeze(0))
#         else:
#             all_sequences_as_vecs.append(torch.stack(current_sequence_embeddings))
#     vecs_padded = pad_sequence(all_sequences_as_vecs, batch_first=False, padding_value=0.0)
#     return vecs_padded, labels_tensor

# def data_loaders_with_glove(
#     train_dataset: pd.DataFrame,
#     test_dataset: pd.DataFrame,
#     validation_dataset: pd.DataFrame,
#     bert_tokenizer: BertTokenizer,
#     batch_size: int = 32,
#     max_seq_len = 512
    
# ):
#     def create_data_list(texts_list, labels_list):
#         return [{'text': text, 'label': label} for text, label in zip(texts_list, labels_list)]

#     train_texts = train_dataset['text_input'].tolist()
#     train_labels = train_dataset['assignee_encoded'].tolist()

#     validation_texts = validation_dataset['text_input'].tolist()
#     validation_labels = validation_dataset['assignee_encoded'].tolist()

#     test_texts = test_dataset['text_input'].tolist()
#     test_labels = test_dataset['assignee_encoded'].tolist()


#     train_data_list = create_data_list(train_texts, train_labels)
#     validation_data_list = create_data_list(validation_texts, validation_labels)
#     test_data_list = create_data_list(test_texts, test_labels)
    
#     vectors_map, unk_embedding = load_glove()

#     collate_fn_custom = partial(collate_with_glove,
#                                 hf_tokenizer=bert_tokenizer,
#                                 glove_word_vectors=vectors_map,
#                                 embedding_dimension=GLOVE_EMBEDDING_DIM,
#                                 unk_word_embedding=unk_embedding,
#                                 max_seq_len=max_seq_len)
#     train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_custom, num_workers=0)
#     validation_loader = DataLoader(validation_data_list, batch_size=batch_size, collate_fn=collate_fn_custom, num_workers=0)
#     test_loader = DataLoader(test_data_list, batch_size=batch_size, collate_fn=collate_fn_custom, num_workers=0)

#     return train_loader, validation_loader, test_loader

## get data loaders second method


def data_loaders_with_glove(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    bert_tokenizer: BertTokenizer,
    batch_size: int = 32,
    remove_stop_words=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training, testing, and validation sets.

    This function initializes custom PyTorch Datasets for each data split and
    then uses a custom collate function to prepare batches with GloVe embeddings.
    """
    print("data_loaders_with_glove 1, 1, 1, 1, 1, 1")

    vectors_map, unk_embedding = load_glove()
    
    print("data_loaders_with_glove 2, 2, 2, 2, 2, 2")

    class TextDataset(Dataset):
        """
        A custom PyTorch Dataset for text data.
        It tokenizes the text using a BERT tokenizer upon initialization.
        """

        def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, remove_stop_words=False):
            """
            Initializes the dataset.
            """
            # Ensure the required columns exist
            if (
                "text_input" not in dataframe.columns
                or "assignee_encoded" not in dataframe.columns
            ):
                raise ValueError(
                    "Input DataFrame must have 'text_input' and 'assignee_encoded' columns."
                )

            self.labels = dataframe["assignee_encoded"].values
            # Tokenize all texts at once for efficiency
            texts = [clean_str(text) for text in dataframe["text_input"]] if remove_stop_words else dataframe["text_input"]
            self.texts = [tokenizer.tokenize(text) for text in  texts]

        def __len__(self) -> int:
            """Returns the number of samples in the dataset."""
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[List[str], int]:
            """
            Retrieves a sample from the dataset.
            """
            return self.texts[idx], self.labels[idx]

    def collate_fn_glove(
        batch: List[Tuple[List[str], int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A custom collate function to process a batch of data for GloVe embeddings.

        This function takes a batch of tokenized texts and labels, converts the tokens
        to their corresponding GloVe vectors, pads the sequences to a uniform length,
        and returns the batch as a pair of PyTorch tensors.
        """
        texts, labels = zip(*batch)

        # Determine the length of the longest sequence in the batch
        max_len = max(len(text) for text in texts)

        # Get the dimension of the GloVe embeddings
        embedding_dim = len(unk_embedding)

        # Create a tensor to hold the padded sequences of embeddings
        padded_texts = torch.zeros(len(texts), max_len, embedding_dim)

        for i, text in enumerate(texts):
            # For each token, look up its embedding, or use the unknown embedding
            embeddings = [
                torch.tensor(vectors_map.get(token, unk_embedding)) for token in text
            ]
            # Stack embeddings for the current text
            if embeddings:
                embeddings_tensor = torch.stack(embeddings)
                padded_texts[i, : len(text)] = embeddings_tensor

        # Convert labels to a tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return padded_texts, labels_tensor

    # Create dataset instances
    train_data = TextDataset(train_dataset, bert_tokenizer, remove_stop_words=remove_stop_words)
    test_data = TextDataset(test_dataset, bert_tokenizer)
    validation_data = TextDataset(validation_dataset, bert_tokenizer)

    # Create DataLoader instances
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_glove
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_glove
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_glove,
    )

    return train_loader, test_loader, validation_loader
