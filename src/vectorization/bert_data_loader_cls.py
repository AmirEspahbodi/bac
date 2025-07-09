import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List
from src.utils import clean_str


def get_cls_embedding(
    text: str, 
    model: BertModel, 
    tokenizer: BertTokenizer, 
    max_length: int = 512,
    device: str = "cpu",
) -> torch.Tensor:
    all_token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunk_size = max_length - 2
    stride = chunk_size // 2
    
    chunks = []
    for i in range(0, len(all_token_ids), stride):
        chunk = all_token_ids[i:i + chunk_size]
        if not chunk:
            break
        chunk_with_specials = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        chunks.append(chunk_with_specials)
        
        if i + chunk_size >= len(all_token_ids):
            break
            
    if not chunks:
        return torch.zeros(model.config.hidden_size, device=device)

    max_chunk_len = max(len(c) for c in chunks)

    input_ids = torch.tensor([c + [tokenizer.pad_token_id] * (max_chunk_len - len(c)) for c in chunks], device=device)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == tokenizer.pad_token_id] = 0

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

    mean_embedding = torch.mean(cls_embeddings, dim=0)
    
    return mean_embedding


def get_data_loaders_bert_cls(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    tokenizer: BertTokenizer,
    batch_size: int = 16,
    remove_stop_words=False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {DEVICE}")

    bert_model_name: str = 'bert-base-uncased'
    print(f"🚀 Loading pre-trained BERT model: '{bert_model_name}'")
    model = BertModel.from_pretrained(bert_model_name).to(DEVICE)
    model.eval()

    def _create_dataset(df: pd.DataFrame, remove_stop_words=False) -> TensorDataset:
        """Helper function to process a dataframe and create a TensorDataset."""
        all_embeddings = []
        all_labels = []

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Embedding {df.name} data"):
            text = str(row['text_input'])
            label = int(row['assignee_encoded'])
            if remove_stop_words:
                text = clean_str(text)
            embedding = get_cls_embedding(text, model, tokenizer, device=DEVICE)
            all_embeddings.append(embedding)
            all_labels.append(label)

        embeddings_tensor = torch.stack(all_embeddings).cpu()
        labels_tensor = torch.tensor(all_labels, dtype=torch.long).cpu()
        
        return TensorDataset(embeddings_tensor, labels_tensor)

    train_dataset.name = 'train'
    validation_dataset.name = 'validation'
    test_dataset.name = 'test'

    print("\n--- Starting Embedding Pre-computation ---")
    
    train_pytorch_dataset = _create_dataset(train_dataset, remove_stop_words=remove_stop_words)
    val_pytorch_dataset = _create_dataset(validation_dataset)
    test_pytorch_dataset = _create_dataset(test_dataset)
    
    print("\n✅ Embedding pre-computation complete.")

    train_loader = DataLoader(
        train_pytorch_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_pytorch_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_pytorch_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    print(f"✅ DataLoaders created with batch size: {batch_size}")
    return train_loader, val_loader, test_loader