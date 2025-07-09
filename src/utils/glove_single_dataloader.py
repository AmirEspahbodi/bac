import torch
from torch.utils.data import DataLoader, TensorDataset

def make_sentence_loader(token_loader: DataLoader,
                         projected_dim: int = None,
                         batch_size: int = 32,
                         shuffle: bool = False,
                         num_workers: int = 0,
                         pin_memory: bool = False) -> DataLoader:
    
    device = next(iter(token_loader))[0].device

    projector = None

    all_means = []
    all_labels = []

    for X_batch, y_batch in token_loader:
        X_mean = X_batch.mean(dim=1)

        if projected_dim is not None:
            if projector is None:
                D_orig = X_mean.size(1)
                projector = torch.nn.Linear(D_orig, projected_dim).to(device)
            X_mean = projector(X_mean)

        all_means.append(X_mean)
        all_labels.append(y_batch)

    X_all = torch.cat(all_means, dim=0)
    y_all = torch.cat(all_labels, dim=0)

    dataset = TensorDataset(X_all, y_all)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)
