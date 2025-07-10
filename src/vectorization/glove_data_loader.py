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
import os
import zipfile
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Union

# Define a type hint for the model for clarity
GloveModel = Dict[str, np.ndarray]

def load_glove_model(dimension: int = 300) -> GloveModel:
    # The cache directory is ~/.glove/
    current_dir = Path(os.getcwd())
    glove_cache_dir = current_dir / ".glove"
    
    # The specific text file we need, e.g., glove.6B.300d.txt
    glove_filename = f"glove.6B.{dimension}d.txt"
    glove_txt_path = glove_cache_dir / glove_filename
    
    # The zip file that contains all dimensions
    glove_zip_filename = "glove.6B.zip"
    glove_zip_path = glove_cache_dir / glove_zip_filename
    
    # The URL for downloading
    glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"

    # --- 2. Check if the final embedding file exists ---
    if glove_txt_path.exists():
        print(f"Found existing GloVe file: '{glove_txt_path}'. Loading into memory.")
        return glove_txt_path

    print(f"GloVe file not found at '{glove_txt_path}'.")

    # --- 3. Ensure the cache directory exists ---
    # This is idempotent; it won't raise an error if the directory already exists.
    try:
        glove_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured cache directory exists: '{glove_cache_dir}'")
    except OSError as e:
        print(f"Error: Could not create directory {glove_cache_dir}. Check permissions.")
        raise IOError(f"Failed to create cache directory: {e}") from e

    # --- 4. Download if the ZIP file doesn't exist ---
    if not glove_zip_path.exists():
        print(f"ZIP archive not found. Downloading from '{glove_url}'...")
        _download_glove(glove_url, glove_zip_path)
    else:
        print(f"Found existing ZIP archive: '{glove_zip_path}'.")

    # --- 5. Extract the ZIP file ---
    print(f"Extracting '{glove_zip_filename}'...")
    try:
        with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
            # Extract all contents to the cache directory
            zip_ref.extractall(glove_cache_dir)
        print("Extraction complete.")
    except zipfile.BadZipFile as e:
        print(f"Error: The downloaded file '{glove_zip_path}' is not a valid ZIP file or is corrupted.")
        print("Please delete the file and try again.")
        raise IOError(f"Bad ZIP file: {e}") from e
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        raise

    # --- 6. Verify extraction and load the model ---
    if not glove_txt_path.exists():
        # This is an important edge case check.
        print(f"Error: Extraction finished, but the required file '{glove_filename}' was not found.")
        raise FileNotFoundError(f"Could not find '{glove_filename}' in the extracted archive.")

    print(f"File is now available at '{glove_txt_path}'. Loading into memory.")
    return glove_txt_path


def _download_glove(url: str, destination_path: Path):
    """Helper function to download a file with a progress bar."""
    try:
        # Use streaming to handle large files efficiently
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit='iB',
                unit_scale=True,
                desc=destination_path.name
            )
            
            with open(destination_path, 'wb') as f:
                for chunk in r.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            
            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("Error: Download was incomplete.")
                raise IOError("Mismatch in downloaded file size.")
                
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        # Clean up partially downloaded file if it exists
        if destination_path.exists():
            os.remove(destination_path)
        raise

    except BaseException as e:
        print(f"Error downloading file: {e}")
        # Clean up partially downloaded file if it exists
        if destination_path.exists():
            os.remove(destination_path)
        raise


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
    glove_txt_path = load_glove_model()

    glove_vectors_map = load_glove_vectors(glove_txt_path, GLOVE_EMBEDDING_DIM)

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
        exit()
    return glove_vectors_map, unk_embedding



def data_loaders_with_glove(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    bert_tokenizer: BertTokenizer,
    batch_size: int = 16,
    remove_stop_words=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training, testing, and validation sets.

    This function initializes custom PyTorch Datasets for each data split and
    then uses a custom collate function to prepare batches with GloVe embeddings.
    """
    print("1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1")

    vectors_map, unk_embedding = load_glove()


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
    print("2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2")
    train_data = TextDataset(train_dataset, bert_tokenizer, remove_stop_words=remove_stop_words)
    print("3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3")
    test_data = TextDataset(test_dataset, bert_tokenizer)
    print("4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4")
    validation_data = TextDataset(validation_dataset, bert_tokenizer)
    print("5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5")

    # Create DataLoader instances
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_glove
    )
    print("6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6")
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_glove
    )
    print("7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7")
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_glove,
    )
    print("8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8")

    return train_loader, test_loader, validation_loader
