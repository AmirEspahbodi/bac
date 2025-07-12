import os
import copy
import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from enum import Enum
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import get_data_loaders
from .models import LSTMModel, LSTMConfig, get_model_info
from .utils import (
    train_one_epoch_m1,
    validation_epoch_fn,
    select_best_optimizer_lr,
)
from src.vectorization import EmbeddingType
from src.dataset._dataset_types import DatasetType


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a lstm model with a specified word embedding type."
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
        choices=[e.value for e in EmbeddingType],
        help="The type of word embedding to use.",
    )

    parser.add_argument(
        "-l",
        "--layers",
        default=1,
        type=int,
        required=False,
        help="Number of LSTM layers.",
    )

    parser.add_argument(
        "-hs",
        "--hidden_size",
        default=128,
        type=int,
        required=False,
        help="Dimention of hidden vector.",
    )
    
    parser.add_argument(
        "-a",
        "--attention",
        default=0,
        type=int,
        required=False,
        choices=[0, 1],
        help="The type of word embedding to use.",
    )
    parser.add_argument(
        "-b",
        "--bidirectional",
        default=0,
        type=int,
        required=False,
        choices=[0, 1],
        help="The type of word embedding to use.",
    )
    parser.add_argument(
        "-r",
        "--residual",
        type=int,
        default=0,
        required=False,
        choices=[0, 1],
        help="residual connection.",
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
    parser.add_argument(
        "-nah",
        "--num_attention_heads",
        type=int,
        default=0,
        required=False,
        choices=[0, 1],
        help="remove stop words on tain dataset.",
    )
    parser.add_argument(
        "-lam",
        "--luong_attention_method",
        type=str,
        default="general",
        required=False,
        choices=["dot", "general", "concat"],
        help="luong attention mechanism.",
    )
    return parser.parse_args()


try:
    args = parse_arguments()
    selected_embedding = EmbeddingType(args.embedding)
    is_bidirectional = args.bidirectional
    is_attention = args.attention
    remove_stop_words = args.remove_stop_words
    num_layers = args.layers
    hidden_size_dim = args.hidden_size
    residual = args.residual
    is_bidirectional =  True if is_bidirectional else False
    is_attention =  True if is_attention else False
    num_attention_heads = args.num_attention_heads
    dataset = args.dataset
    NUM_EPOCHS = args.num_epochs
    luong_attention_method = args.luong_attention_method


except argparse.ArgumentError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

aug_train_loader, val_loader, test_loader, NUM_ACTUAL_CLS = get_data_loaders(
    dataset, selected_embedding, remove_stop_words=remove_stop_words
)

Path(f"{os.getcwd()}/.models").mkdir(parents=True, exist_ok=True)
Path(f"{os.getcwd()}/.result").mkdir(parents=True, exist_ok=True)


# --- Model & Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
match selected_embedding:
    case EmbeddingType.GLOVE:
        print("using  GLOVE embeddings")
        config = LSTMConfig(
            hidden_size=hidden_size_dim,
            num_layers=num_layers,
            dropout=0.4,
            input_size = 300,
            num_classes = NUM_ACTUAL_CLS,
            residual_connections=residual,
            bidirectional=is_bidirectional,
            use_attention=is_attention,
            attention_heads=num_attention_heads,
            luong_attention_method=luong_attention_method
        )
        model_save_path = f"{os.getcwd()}/.models/lstm_glove_model.pt"
        result_save_path = f"{os.getcwd()}/.result/lstm_glove_result.json"


    case EmbeddingType.BERT:
        print("using bert embeddings")
        config = LSTMConfig(
            hidden_size=hidden_size_dim,
            num_layers=num_layers,
            dropout=0.4,
            input_size = 768,
            num_classes = NUM_ACTUAL_CLS,
            residual_connections=residual,
            bidirectional=is_bidirectional,
            use_attention=is_attention,
            attention_heads=num_attention_heads,
            luong_attention_method=luong_attention_method
        )
        model_save_path = f"{os.getcwd()}/.models/lstm_bert_model.pt"
        result_save_path = f"{os.getcwd()}/.result/lstm_bert_result.json"

lstm_model = LSTMModel(config).to(device=DEVICE)

print(f"\n------\nlstm model info: \n{get_model_info(lstm_model)}\n------\n")

LABEL_SMOOTHING_FACTOR = 0.1
GRADIENT_CLIP_VALUE = 1.0


# --- Early Stopping Configuration ---
PATIENCE = 5
MIN_DELTA = 0.0001



loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_FACTOR)


selected_optimizer_class, selected_lr = select_best_optimizer_lr(
    1,
    lstm_model,
    aug_train_loader,
    loss_fn,
    GRADIENT_CLIP_VALUE,
    DEVICE
)

print(selected_optimizer_class, selected_lr)

if selected_optimizer_class is optim.SGD:
    WEIGHT_DECAY = 0.0001
    optimizer = selected_optimizer_class(
        lstm_model.parameters(), lr=selected_lr, weight_decay=WEIGHT_DECAY, momentum=0.9
    )

else:
    if selected_optimizer_class is optim.AdamW:
        WEIGHT_DECAY = 0.01
    elif selected_optimizer_class is optim.Adam:
        WEIGHT_DECAY = 0.0001
        
    optimizer = selected_optimizer_class(
        lstm_model.parameters(),
        lr=selected_lr,
        weight_decay=WEIGHT_DECAY,
    )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)

loss_train_hist = []
loss_valid_hist = []
acc_train_hist = []
acc_valid_hist = []


best_val_loss = float("inf")
epochs_no_improve = 0
early_stop_triggered = False

best_model_state = None


print(f"\n✅ Start Training {NUM_EPOCHS} epochs ...")
for epoch in range(NUM_EPOCHS):
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | Current LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # --- Train for one epoch ---
    lstm_model, loss_train, acc_train = train_one_epoch_m1(
        lstm_model,
        aug_train_loader,
        loss_fn,
        optimizer,
        DEVICE,
        GRADIENT_CLIP_VALUE,
        epoch=epoch + 1,
    )

    # --- Validate for one epoch ---
    current_val_loss, current_val_acc, _, _ = validation_epoch_fn(
        lstm_model, val_loader, loss_fn, DEVICE, description=f"Validation {epoch + 1}"
    )

    # --- Update history ---
    loss_train_hist.append(loss_train)
    loss_valid_hist.append(current_val_loss)
    acc_train_hist.append(acc_train)
    acc_valid_hist.append(current_val_acc)

    print(f"\tTrain: Loss = {loss_train:.4f}, Acc = {acc_train:.4f}")
    print(f"\tValid: Loss = {current_val_loss:.4f}, Acc = {current_val_acc:.4f}")

    # --- Early Stopping and Model Checkpointing ---
    if current_val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = current_val_loss
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(lstm_model.state_dict())
        print(f"\t✨ New best validation loss: {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(
            f"\tValidation loss did not improve. Patience: {epochs_no_improve}/{PATIENCE}"
        )

    # --- Step the scheduler ---
    scheduler.step()

    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}.")
        NUM_EPOCHS = epoch + 1
        break

    print("-" * 60)


if best_model_state:
    lstm_model.load_state_dict(best_model_state)
    torch.save(lstm_model, model_save_path)
    print(f"\n✅ Loaded best model based on validation accuracy for final testing. {model_save_path}")
else:
    print("\n⚠️ No improvement observed. Using model from the last epoch for testing.")


epochs_range = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss_train_hist, "bo-", label="Training Loss")
plt.plot(epochs_range, loss_valid_hist, "ro-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc_train_hist, "bo-", label="Training Accuracy (Top-1)")
plt.plot(epochs_range, acc_valid_hist, "ro-", label="Validation Accuracy (Top-1)")
plt.title("Training and Validation Accuracy (Top-1)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\n🧪 Evaluating on Test Set...")
test_loss_final, test_acc_top1_final, test_mrr_final, _ = (
    test_loss_final,
    test_acc_top1_final,
    test_mrr_final,
    _,
) = validation_epoch_fn(lstm_model, test_loader, loss_fn, DEVICE, description=f"Testing")

print(f"\n--- Test Set Results (Best Validation Model) ---")
print(f"\tTest Loss: {test_loss_final:.4f}")
print(f"\tTest Accuracy (Top-1): {test_acc_top1_final:.2f}%")
print(f"\tTest MRR: {test_mrr_final:.4f}")


k_values_for_test = [5, 10]

_, _, _, test_top_5_acc = validation_epoch_fn(
    lstm_model, test_loader, loss_fn, DEVICE, f"Testing (Top-{5})", k_for_top_k_eval=5
)
_, _, _, test_top_10_acc = validation_epoch_fn(
    lstm_model, test_loader, loss_fn, DEVICE, f"Testing (Top-{10})", k_for_top_k_eval=10
)

result = {
    "loss_train_hist": loss_train_hist,
    "loss_valid_hist": loss_valid_hist,
    "acc_train_hist": acc_train_hist,
    "acc_valid_hist": acc_valid_hist,
    "test_top_5_acc": test_top_5_acc,
    "test_top_10_acc": test_top_10_acc,
    "test_loss_final": test_loss_final,
    "test_acc_top1_final": test_acc_top1_final,
    "test_mrr_final": test_mrr_final,
}
print(result)

# Save to JSON file
with open(result_save_path, "w") as f:
    json.dump(result, f, indent=4)
