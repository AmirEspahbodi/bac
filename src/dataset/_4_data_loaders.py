import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Any
from transformers import BertTokenizer


def create_data_loaders_word_embedding(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    bert_tokenizer: BertTokenizer,
    vectorization_function,
    batch_size: int = 32,
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

    def collate_fn_glove_w2v(
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
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_glove_w2v
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_glove_w2v
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_glove_w2v,
    )

    return train_loader, test_loader, validation_loader


class ContextualizedTextDataset(Dataset):
    """
    Custom Dataset for handling text and labels for contextualized models like BERT.
    It takes a dataframe, a tokenizer, the column name for text, and the column name for labels.
    """

    def __init__(self, dataframe, tokenizer, text_col, label_col, max_len=128):
        """
        Args:
            dataframe (pd.DataFrame): The dataset containing text and labels.
            tokenizer: The tokenizer to use (e.g., BertTokenizer).
            text_col (str): The name of the column containing the text.
            label_col (str): The name of the column containing the encoded labels.
            max_len (int): The maximum sequence length for the tokenizer.
        """
        # Ensure the required columns exist
        if text_col not in dataframe.columns or label_col not in dataframe.columns:
            raise ValueError(f"Dataframe must contain '{text_col}' and '{label_col}' columns.")

        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe[text_col]
        self.labels = dataframe[label_col]
        self.max_len = max_len

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset and tokenizes it.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - text (str): The original text.
                - input_ids (torch.Tensor): The token IDs.
                - attention_mask (torch.Tensor): The attention mask.
                - labels (torch.Tensor): The label of the text.
        """
        text = str(self.text.iloc[index])
        label = self.labels.iloc[index]

        # Tokenize the text using the BERT tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,      # Pad & truncate
            padding='max_length',         # Pad to max_length
            truncation=True,              # Truncate to max_length
            return_attention_mask=True,
            return_tensors='pt',          # Return PyTorch tensors
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# --- 2. The Main Function to Create DataLoaders ---

def create_data_loaders_contextualized(
    train_dataset,
    validation_dataset,
    test_dataset,
    bert_tokenizer,
    batch_size=32,
    max_len=128
):
    """
    Creates training, validation, and test DataLoaders for contextualized models.

    A note on ELMo vs. BERT:
    This function is optimized for tokenizers like BERT's. ELMo embeddings are
    character-based and don't use a WordPiece tokenizer. To use ELMo, you would
    typically use a library like AllenNLP, which has its own data loading and
    batching mechanisms. You would need a different `Dataset` class that processes
    text into character IDs for the ELMo model. This implementation focuses on the
    more common transformer-based approach.

    Args:
        train_dataset (pd.DataFrame): DataFrame for training.
        validation_dataset (pd.DataFrame): DataFrame for validation.
        test_dataset (pd.DataFrame): DataFrame for testing.
        bert_tokenizer: An instance of a Hugging Face tokenizer (e.g., BertTokenizer).
        batch_size (int, optional): The batch size for the DataLoaders. Defaults to 32.
        max_len (int, optional): Maximum sequence length. Defaults to 128.

    Returns:
        A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
    """
    # Create instances of our custom Dataset
    train_ds = ContextualizedTextDataset(
        dataframe=train_dataset,
        tokenizer=bert_tokenizer,
        text_col='text_input',
        label_col='assignee_encoded',
        max_len=max_len
    )

    val_ds = ContextualizedTextDataset(
        dataframe=validation_dataset,
        tokenizer=bert_tokenizer,
        text_col='text_input',
        label_col='assignee_encoded',
        max_len=max_len
    )

    test_ds = ContextualizedTextDataset(
        dataframe=test_dataset,
        tokenizer=bert_tokenizer,
        text_col='text_input',
        label_col='assignee_encoded',
        max_len=max_len
    )

    # Create the DataLoaders
    # The `collate_fn` is handled implicitly by the Hugging Face tokenizer
    # when `return_tensors='pt'` is used and items are batched.
    # We set `num_workers` for potentially faster data loading.
    aug_train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data
        num_workers=2
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=2
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=2
    )

    print(f"DataLoaders created with batch size: {batch_size}")
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    return aug_train_loader, val_loader, test_loader