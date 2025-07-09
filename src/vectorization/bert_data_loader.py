import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List
from src.utils import clean_str

# Ensure you have the necessary libraries installed:
# pip install torch pandas tqdm transformers

def _create_full_sequence_embeddings(
    texts: pd.Series,
    model: BertModel,
    tokenizer: BertTokenizer,
    device: torch.device,
    max_len: int = 512,
    batch_size: int = 16,
    remove_stop_words=False
) -> torch.Tensor:
    """
    Generates embeddings for a series of texts using a sliding window approach for long documents.
    """
    all_embeddings: List[torch.Tensor] = []
    model.eval()

    # Process texts in batches for significant speed improvement
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Full Embeddings"):
        # Select a batch of texts from the pandas Series
        batch_texts = texts.iloc[i:i + batch_size].tolist()
        if remove_stop_words:
            batch_texts = [clean_str(s) for s in batch_texts]

        # Tokenize the batch. The tokenizer handles truncation and padding for you.
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",  # Pad shorter sentences to max_len
            truncation=True,       # Truncate longer sentences to max_len
            max_length=max_len
        )

        # Move tokenized inputs to the specified device (GPU/CPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference without calculating gradients
        with torch.no_grad():
            output = model(**inputs)

        # output.last_hidden_state contains the embeddings for all tokens in the sequence
        # Its shape is (batch_size, max_len, embedding_dim)
        last_hidden_states = output.last_hidden_state
        
        # Move embeddings to CPU and append to our list
        all_embeddings.append(last_hidden_states.cpu())

    # Concatenate all batch results into a single final tensor
    finall_embeddings = torch.cat(all_embeddings, dim=0)
    return finall_embeddings


def get_data_loaders_bert(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    tokenizer: BertTokenizer,
    batch_size: int = 16,
    remove_stop_words=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for training, validation, and test sets.
    """
    # --- 1. Setup Model and Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    bert_model.eval() # Set model to evaluation mode

    # --- 2. Generate Embeddings for each dataset split ---
    train_embeddings = _create_full_sequence_embeddings(
        train_dataset['text_input'], bert_model, tokenizer, device, remove_stop_words=remove_stop_words
    )
    val_embeddings = _create_full_sequence_embeddings(
        validation_dataset['text_input'], bert_model, tokenizer, device
    )
    test_embeddings = _create_full_sequence_embeddings(
        test_dataset['text_input'], bert_model, tokenizer, device
    )

    # --- 3. Get Labels ---
    train_labels = torch.tensor(train_dataset['assignee_encoded'].values, dtype=torch.long)
    val_labels = torch.tensor(validation_dataset['assignee_encoded'].values, dtype=torch.long)
    test_labels = torch.tensor(test_dataset['assignee_encoded'].values, dtype=torch.long)

    # --- 4. Create TensorDatasets ---
    # A TensorDataset is an efficient way to wrap data and targets as a PyTorch Dataset
    train_data = TensorDataset(train_embeddings, train_labels)
    val_data = TensorDataset(val_embeddings, val_labels)
    test_data = TensorDataset(test_embeddings, test_labels)

    # --- 5. Create DataLoaders ---
    # The DataLoader provides an iterable over the dataset, with support for
    # batching, shuffling, and multiprocessing.
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data to ensure model generalization
        num_workers=2 # Use multiple subprocesses for data loading
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

    print("\nDataLoaders created successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader
