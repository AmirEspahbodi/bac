import torch
from torch.utils.data import DataLoader, TensorDataset

def single_vector_glove_dataloader(token_loader: DataLoader,
                         batch_size: int = 32,
                         shuffle: bool = False,
                         num_workers: int = 0,
                         pin_memory: bool = False) -> DataLoader:

    all_means = []
    all_labels = []

    for X_batch, y_batch in token_loader:
        X_mean = X_batch.mean(dim=1)
        all_means.append(X_mean)
        all_labels.append(y_batch)

    X_all = torch.cat(all_means, dim=0)
    y_all = torch.cat(all_labels, dim=0)

    dataset = TensorDataset(X_all, y_all)
    new_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    return new_loader
