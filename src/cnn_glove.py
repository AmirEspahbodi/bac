import torch
import torch.nn as nn
from .dataset import get_data_loaders
from .models import CNNModel
from .utils import (
    AverageMeter,
    calculate_mrr_fn,
    plot_training_history,
    top_k_accuracy_fn,
    train_one_epoch,
    validation,
    select_best_optimizer_lr,
)


# --- Model & Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
EMBEDDING_DIM_VALUE = 300
N_FILTERS_LIST = [512, 512, 512]
FILTER_SIZES_LIST = [3, 4, 5]
DROPOUT_RATE_VALUE = 0.5
HIDDEN_DIM_FC_VALUE = 256

aug_train_loader, val_loader, test_loader, NUM_ACTUAL_CLS = get_data_loaders()
loss_fn = nn.CrossEntropyLoss()

cnn_model = CNNModel(
    embed_dim=EMBEDDING_DIM_VALUE,
    filter_sizes=FILTER_SIZES_LIST,
    num_filters_per_size=N_FILTERS_LIST,
    num_classes=NUM_ACTUAL_CLS,
    dropout_rate=DROPOUT_RATE_VALUE,
    hidden_dim_fc=HIDDEN_DIM_FC_VALUE,
).to(DEVICE)


selected_optimizer, selected_lr = select_best_optimizer_lr(
    3,
    cnn_model,
    aug_train_loader,
    loss_fn,
    DEVICE
)

print(selected_optimizer, selected_lr)

print(len(next(iter(aug_train_loader))))