import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Any

def create_data_loaders(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    bert_tokenizer: BertTokenizer,
    vectorization_function,
    
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training, testing, and validation sets.

    This function initializes custom PyTorch Datasets for each data split and
    then uses a custom collate function to prepare batches with GloVe embeddings.
    """

    vectors_map, unk_embedding = vectorization_function()
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
            if 'text_input' not in dataframe.columns or 'assignee_encoded' not in dataframe.columns:
                raise ValueError("Input DataFrame must have 'text_input' and 'assignee_encoded' columns.")
                
            self.labels = dataframe['assignee_encoded'].values
            # Tokenize all texts at once for efficiency
            self.texts = [tokenizer.tokenize(text) for text in dataframe['text_input']]

        def __len__(self) -> int:
            """Returns the number of samples in the dataset."""
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[List[str], int]:
            """
            Retrieves a sample from the dataset.
            """
            return self.texts[idx], self.labels[idx]

    def collate_fn_glove(batch: List[Tuple[List[str], int]]) -> Tuple[torch.Tensor, torch.Tensor]:
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
            embeddings = [torch.tensor(vectors_map.get(token, unk_embedding)) for token in text]
            # Stack embeddings for the current text
            if embeddings:
                embeddings_tensor = torch.stack(embeddings)
                padded_texts[i, :len(text)] = embeddings_tensor

        # Convert labels to a tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return padded_texts, labels_tensor

    # Create dataset instances
    train_data = TextDataset(train_dataset, bert_tokenizer)
    test_data = TextDataset(test_dataset, bert_tokenizer)
    validation_data = TextDataset(validation_dataset, bert_tokenizer)

    # Create DataLoader instances
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_glove
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_glove
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_glove
    )

    return train_loader, test_loader, validation_loader