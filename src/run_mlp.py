import argparse
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .dataset import get_data_loaders
from src.vectorization import EmbeddingType
from src.dataset._dataset_types import DatasetType
from src.models import MLPClassifier
from .utils import (
    train_one_epoch_m1,
    validation_epoch_fn,
    select_best_optimizer_lr,
    single_vector_glove_dataloader
)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a CNN model with a specified word embedding type."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=[e.value for e in DatasetType],
        help="dataset type.",
    )
    parser.add_argument(
        "-ne",
        "--num_epochs",
        type=int,
        required=True,
        choices=[*range(0, 20)],
        help="dataset type.",
    )
    
    parser.add_argument(
        "-e",
        "--embedding",
        type=str,
        required=True,
        choices=["glove", "bert_cls"],
        help="The type of word embedding to use.",
    )

    parser.add_argument(
        "-rsw",
        "--remove_stop_words",
        type=int,
        default=0,
        required=False,
        choices=[0, 1],
        help="remove stop words on tain dataset.",
    )

    return parser.parse_args()


try:
    args = parse_arguments()
    selected_embedding = EmbeddingType(args.embedding)
    remove_stop_words = args.remove_stop_words
    dataset = args.dataset
    NUM_EPOCHS = args.num_epochs

except argparse.ArgumentError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

aug_train_loader, val_loader, test_loader, NUM_LABELS = get_data_loaders(
    dataset, selected_embedding, remove_stop_words
)

print(selected_embedding)

if EmbeddingType.GLOVE == selected_embedding:
    aug_train_loader = single_vector_glove_dataloader(aug_train_loader)
    val_loader = single_vector_glove_dataloader(val_loader)
    test_loader = single_vector_glove_dataloader(test_loader)


BERT_DIM = 768
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 512
NUM_BLOCKS = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1


LABEL_SMOOTHING_FACTOR = 0.1
GRADIENT_CLIP_VALUE = 1.0


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


mlp_model = MLPClassifier(
    input_dim=BERT_DIM, 
    hidden_dim=HIDDEN_DIM, 
    output_dim=NUM_LABELS, 
    num_blocks=NUM_BLOCKS, 
    dropout_rate=DROPOUT
)

loss_fn = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING_FACTOR)

selected_optimizer_class, selected_lr = select_best_optimizer_lr(
    1,
    mlp_model,
    aug_train_loader,
    loss_fn,
    GRADIENT_CLIP_VALUE,
    DEVICE
)

print(selected_optimizer_class, selected_lr)

if selected_optimizer_class is optim.SGD:
    WEIGHT_DECAY = 0.0001
    optimizer = selected_optimizer_class(
        mlp_model.parameters(), lr=selected_lr, weight_decay=WEIGHT_DECAY, momentum=0.9
    )

else:
    if selected_optimizer_class is optim.AdamW:
        WEIGHT_DECAY = 0.01
    elif selected_optimizer_class is optim.Adam:
        WEIGHT_DECAY = 0.0001
        
    optimizer = selected_optimizer_class(
        mlp_model.parameters(),
        lr=selected_lr,
        weight_decay=WEIGHT_DECAY,
    )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)
