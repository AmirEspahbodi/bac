import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any
from src.vectorization import EmbeddingType

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


####################################################################################################################

class PandasTextDataset(Dataset):
    """
    A custom PyTorch Dataset that wraps a pandas DataFrame.
    """
    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if 'text_input' not in dataframe.columns or 'assignee_encoded' not in dataframe.columns:
            raise ValueError("DataFrame must contain 'text_input' and 'assignee_encoded' columns.")
        
        self.df = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        """Returns the number of rows in the DataFrame."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the DataFrame at the specified index.
        """
        row = self.df.iloc[idx]
        return {
            'text_input': row['text_input'],
            'assignee_encoded': row['assignee_encoded']
        }

class ModernEmbeddingCollator:
    def __init__(self,
                 bert_tokenizer: BertTokenizer,
                 embedding_type: EmbeddingType,
                 bert_model_name: str = 'bert-base-uncased',
                 st_model_name: str = 'all-MiniLM-L6-v2',
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.bert_tokenizer = bert_tokenizer
        self.embedding_type=embedding_type

        print("Initializing embedding models...")
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device)
        self.st_model = SentenceTransformer(st_model_name, device=self.device)
        self.bert_model.eval()
        self.st_model.eval()
        print(f"Models loaded and moved to {self.device}.")

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        texts: List[str] = [item['text_input'] for item in batch]
        labels: torch.Tensor = torch.tensor(
            [item['assignee_encoded'] for item in batch],
            dtype=torch.long
        ).to(self.device)

        with torch.no_grad():
            match self.embedding_type:
                case EmbeddingType.BERT:
                    bert_inputs = self.bert_tokenizer(
                        texts, padding='longest', truncation=True, max_length=512, return_tensors='pt'
                    )
                    bert_inputs = {k: v.to(self.device) for k, v in bert_inputs.items()}
                    bert_outputs = self.bert_model(**bert_inputs)
                    embeddings = bert_outputs.last_hidden_state
                case EmbeddingType.ST:
                    embeddings = self.st_model.encode(
                        texts, convert_to_tensor=True, device=self.device
                    )

            # combined_embeddings = torch.cat((bert_cls_embeddings, st_embeddings), dim=1)

        return embeddings, labels

def create_data_loaders_contectualized(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bert_tokenizer: BertTokenizer,
    embedding_type: EmbeddingType, 
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders from pandas DataFrames with on-the-fly embedding generation.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Step 1: Instantiate the custom collator
    collator = ModernEmbeddingCollator(bert_tokenizer=bert_tokenizer, device=device, embedding_type=embedding_type)

    # Step 2: Wrap DataFrames in the custom PyTorch Dataset class
    train_torch_dataset = PandasTextDataset(train_df)
    val_torch_dataset = PandasTextDataset(validation_df)
    test_torch_dataset = PandasTextDataset(test_df)

    # Step 3: Create DataLoaders using the wrapped datasets and the collator
    train_loader = DataLoader(
        dataset=train_torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        dataset=val_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        dataset=test_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader
