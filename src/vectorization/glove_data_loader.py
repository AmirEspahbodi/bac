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


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- GloVe Configuration ---
GLOVE_DIR = ".cache/glove"
GLOVE_FILE_NAME = "glove.6B.300d.txt"
GLOVE_EMBEDDING_DIM = 300
GLOVE_PATH = f"./{GLOVE_DIR}/{GLOVE_FILE_NAME}"
GLOVE_ZIP_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_LOCAL_ZIP_PATH = f"{GLOVE_DIR}/glove.6B.zip"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- GloVe File Handling ---
def ensure_glove_file_is_present():
    if os.path.exists(GLOVE_PATH):
        logging.info(f"Found existing GloVe text file: {GLOVE_PATH}")
        return True
    if not os.path.exists(GLOVE_LOCAL_ZIP_PATH):
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
            logging.error(f"Error downloading GloVe zip file: {e}")
            if os.path.exists(GLOVE_LOCAL_ZIP_PATH):
                os.remove(GLOVE_LOCAL_ZIP_PATH)
            return False
        except Exception as e:
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
        with zipfile.ZipFile(GLOVE_LOCAL_ZIP_PATH, "r") as zip_ref:
            if GLOVE_FILE_NAME in zip_ref.namelist():
                zip_ref.extract(
                    GLOVE_FILE_NAME, path=os.path.dirname(GLOVE_PATH) or "."
                )
                logging.info(
                    f"Successfully extracted {GLOVE_FILE_NAME} to {GLOVE_PATH}"
                )
                return True
            else:
                logging.error(
                    f"Error: {GLOVE_FILE_NAME} not found inside {GLOVE_LOCAL_ZIP_PATH}."
                )
                logging.info(f"Available files: {zip_ref.namelist()}")
                return False
    except zipfile.BadZipFile:
        logging.error(
            f"Error: {GLOVE_LOCAL_ZIP_PATH} is a bad zip file. Please delete it and try again."
        )
        return False
    except Exception as e:
        logging.error(f"An error occurred during extraction: {e}")
        return False


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
    if not ensure_glove_file_is_present():
        logging.error("Could not obtain GloVe file. Exiting.")
        exit()

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


def data_loaders_with_glove(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    bert_tokenizer: BertTokenizer,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training, testing, and validation sets.

    This function initializes custom PyTorch Datasets for each data split and
    then uses a custom collate function to prepare batches with GloVe embeddings.
    """

    vectors_map, unk_embedding = load_glove()

    class TextDataset(Dataset):
        """
        A custom PyTorch Dataset for text data.
        It tokenizes the text using a BERT tokenizer upon initialization.
        """

        def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer):
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
            self.texts = [tokenizer.tokenize(text) for text in dataframe["text_input"]]

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
    train_data = TextDataset(train_dataset, bert_tokenizer)
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
