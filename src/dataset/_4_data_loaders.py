import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer
import numpy as np
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuration class for data loader parameters."""

    batch_size: int = 32
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    max_length: int = 512
    text_column: str = "text_input"
    label_column: str = "assignee_encoded"


class TextDataset(Dataset):
    """
    Custom Dataset class for handling textual data with both BERT and GloVe processing.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        bert_tokenizer: AutoTokenizer,
        vectorization_vectors_map: Dict[str, np.ndarray],
        unk_embedding: np.ndarray,
        config: DataLoaderConfig,
        is_training: bool = False,
    ):
        self.df = dataframe.copy()
        self.bert_tokenizer = bert_tokenizer
        self.vectorization_vectors_map = vectorization_vectors_map
        self.unk_embedding = unk_embedding
        self.config = config
        self.is_training = is_training

        # Validate required columns
        if config.text_column not in self.df.columns:
            raise ValueError(
                f"Text column '{config.text_column}' not found in DataFrame"
            )
        if config.label_column not in self.df.columns:
            raise ValueError(
                f"Label column '{config.label_column}' not found in DataFrame"
            )

        # Clean and preprocess
        self._preprocess_data()

        logger.info(f"Initialized TextDataset with {len(self.df)} samples")

    def _preprocess_data(self) -> None:
        """Preprocess the DataFrame by handling missing values and basic cleaning."""
        # Handle missing values
        self.df[self.config.text_column] = self.df[self.config.text_column].fillna("")

        # Remove empty texts
        initial_len = len(self.df)
        self.df = self.df[self.df[self.config.text_column].str.strip() != ""]
        final_len = len(self.df)

        if initial_len != final_len:
            logger.warning(f"Removed {initial_len - final_len} empty text samples")

        # Reset index
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get text and label
        text = str(self.df.iloc[idx][self.config.text_column])
        label = self.df.iloc[idx][self.config.label_column]

        # BERT tokenization
        bert_encoding = self.bert_tokenizer(
            text,
            truncation=True,
            padding=False,  # Will be handled by collate function
            max_length=self.config.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # GloVe processing
        glove_embeddings = self._text_to_glove_embeddings(text)

        return {
            "text": text,
            "bert_input_ids": bert_encoding["input_ids"].squeeze(0),
            "bert_attention_mask": bert_encoding["attention_mask"].squeeze(0),
            "glove_embeddings": torch.FloatTensor(glove_embeddings),
            "label": torch.tensor(label, dtype=torch.long),
            "index": idx,
        }

    def _text_to_glove_embeddings(self, text: str) -> np.ndarray:
        """
        Convert text to GloVe embeddings.
        """
        # Simple tokenization (can be enhanced with spaCy or NLTK)
        tokens = text.lower().split()

        # Limit sequence length for GloVe (keeping some room for special tokens)
        max_glove_len = min(self.config.max_length - 2, len(tokens))
        tokens = tokens[:max_glove_len]

        embeddings = []
        for token in tokens:
            if token in self.vectorization_vectors_map:
                embeddings.append(self.vectorization_vectors_map[token])
            else:
                embeddings.append(self.unk_embedding)

        # Handle empty sequences
        if not embeddings:
            embeddings = [self.unk_embedding]

        return np.array(embeddings)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples with both BERT and GloVe features.
    """
    # Extract components
    texts = [item["text"] for item in batch]
    bert_input_ids = [item["bert_input_ids"] for item in batch]
    bert_attention_masks = [item["bert_attention_mask"] for item in batch]
    glove_embeddings = [item["glove_embeddings"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    indices = torch.tensor([item["index"] for item in batch])

    # Pad BERT sequences
    bert_input_ids_padded = pad_sequence(
        bert_input_ids, batch_first=True, padding_value=0
    )
    bert_attention_masks_padded = pad_sequence(
        bert_attention_masks, batch_first=True, padding_value=0
    )

    # Pad GloVe embeddings
    glove_embeddings_padded = pad_sequence(
        glove_embeddings, batch_first=True, padding_value=0.0
    )

    # Create attention mask for GloVe embeddings
    glove_attention_mask = torch.zeros(
        glove_embeddings_padded.shape[:2], dtype=torch.bool
    )
    for i, emb in enumerate(glove_embeddings):
        glove_attention_mask[i, : len(emb)] = True

    return {
        "texts": texts,
        "bert_input_ids": bert_input_ids_padded,
        "bert_attention_mask": bert_attention_masks_padded,
        "glove_embeddings": glove_embeddings_padded,
        "glove_attention_mask": glove_attention_mask,
        "labels": labels,
        "indices": indices,
        "batch_size": len(batch),
    }


def create_data_loaders(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    bert_tokenizer: AutoTokenizer,
    vectorization_function,
    train_config: Optional[DataLoaderConfig] = None,
    eval_config: Optional[DataLoaderConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders with different configurations for training and evaluation.
    """
    vectorization_vectors_map, unk_embedding = vectorization_function()

    if train_config is None:
        train_config = DataLoaderConfig(
            batch_size=32, shuffle_train=True, drop_last=True
        )

    if eval_config is None:
        eval_config = DataLoaderConfig(
            batch_size=64,  # Larger batch size for evaluation
            shuffle_train=False,
            shuffle_val=False,
            shuffle_test=False,
            drop_last=False,
        )

    train_ds = TextDataset(
        train_dataset,
        bert_tokenizer,
        vectorization_vectors_map,
        unk_embedding,
        train_config,
        is_training=True,
    )
    val_ds = TextDataset(
        validation_dataset,
        bert_tokenizer,
        vectorization_vectors_map,
        unk_embedding,
        eval_config,
        is_training=False,
    )
    test_ds = TextDataset(
        test_dataset,
        bert_tokenizer,
        vectorization_vectors_map,
        unk_embedding,
        eval_config,
        is_training=False,
    )

    # Create training loader
    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=train_config.shuffle_train,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        drop_last=train_config.drop_last,
        collate_fn=collate_fn,
    )

    # Create validation loader
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_config.batch_size,
        shuffle=eval_config.shuffle_val,
        num_workers=eval_config.num_workers,
        pin_memory=eval_config.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Create test loader
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_config.batch_size,
        shuffle=eval_config.shuffle_test,
        num_workers=eval_config.num_workers,
        pin_memory=eval_config.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
