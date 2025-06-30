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
from .models import CNNModelGLOVE, CNNModelBERT
from .utils import (
    train_one_epoch_m1,
    validation_epoch_fn,
    select_best_optimizer_lr,
)
from src.vectorization import EmbeddingType


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a CNN model with a specified word embedding type."
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

except argparse.ArgumentError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

aug_train_loader, val_loader, test_loader, NUM_ACTUAL_CLS = get_data_loaders(
    selected_embedding, remove_stop_words
)

# --- Model & Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
match selected_embedding:
    case EmbeddingType.GLOVE:
        print("using  GLOVE embeddings")
        NUM_EPOCHS = 7
        EMBEDDING_DIM_VALUE = 300
        N_FILTERS_LIST = [256, 256, 256]
        FILTER_SIZES_LIST = [3, 4, 5]
        OUTPUT_DIM_VALUE = NUM_ACTUAL_CLS
        DROPOUT_RATE_VALUE = 0.5
        HIDDEN_DIM_FC_VALUE = 128
        cnn_model = CNNModelGLOVE(
            embed_dim=EMBEDDING_DIM_VALUE,
            filter_sizes=FILTER_SIZES_LIST,
            num_filters_per_size=N_FILTERS_LIST,
            num_classes=OUTPUT_DIM_VALUE,
            dropout_rate=DROPOUT_RATE_VALUE,
            hidden_dim_fc=HIDDEN_DIM_FC_VALUE
        ).to(DEVICE)
        BATCH_SIZE = 32
        model_save_path = f"{os.getcwd()}/.models/cnn_glove_model.pt"
        result_save_path = f"{os.getcwd()}/.result/cnn_glove_result.json"
        Path(f"{os.getcwd()}/.models").mkdir(parents=True, exist_ok=True)
        Path(f"{os.getcwd()}/.result").mkdir(parents=True, exist_ok=True)

    case EmbeddingType.BERT:
        print("using bert embeddings")
        # NUM_EPOCHS = 7
        # EMBEDDING_DIM_VALUE = 768
        # N_FILTERS_LIST = [128, 128, 128, 128]
        # FILTER_SIZES_LIST = [2, 3, 4, 5]
        # DROPOUT_RATE_VALUE = 0.5
        # HIDDEN_DIM_FC1_VALUE = 256
        # HIDDEN_DIM_FC2_VALUE = 128
        # cnn_model = CNNModelBERT(
        #     embed_dim=EMBEDDING_DIM_VALUE,
        #     filter_sizes=FILTER_SIZES_LIST,
        #     num_filters_per_size=N_FILTERS_LIST,
        #     num_classes=NUM_ACTUAL_CLS,
        #     dropout_rate=DROPOUT_RATE_VALUE,
        #     hidden_dim_fc1=HIDDEN_DIM_FC1_VALUE,
        #     hidden_dim_fc2=HIDDEN_DIM_FC2_VALUE,
        # ).to(DEVICE)
        # BATCH_SIZE = 32
        NUM_EPOCHS = 7
        EMBEDDING_DIM_VALUE = 768
        N_FILTERS_LIST = [256, 256, 256]
        FILTER_SIZES_LIST = [3, 4, 5]
        OUTPUT_DIM_VALUE = NUM_ACTUAL_CLS
        DROPOUT_RATE_VALUE = 0.5
        HIDDEN_DIM_FC_VALUE = 128
        cnn_model = CNNModelGLOVE(
            embed_dim=EMBEDDING_DIM_VALUE,
            filter_sizes=FILTER_SIZES_LIST,
            num_filters_per_size=N_FILTERS_LIST,
            num_classes=OUTPUT_DIM_VALUE,
            dropout_rate=DROPOUT_RATE_VALUE,
            hidden_dim_fc=HIDDEN_DIM_FC_VALUE
        ).to(DEVICE)
        BATCH_SIZE = 32
        model_save_path = f"{os.getcwd()}/.models/cnn_bert_model.pt"
        result_save_path = f"{os.getcwd()}/.result/cnn_bert_result.json"
        Path(f"{os.getcwd()}/.models").mkdir(parents=True, exist_ok=True)
        Path(f"{os.getcwd()}/.result").mkdir(parents=True, exist_ok=True)

LABEL_SMOOTHING_FACTOR = 0.1
GRADIENT_CLIP_VALUE = 1.0


# --- Early Stopping Configuration ---
PATIENCE = 5
MIN_DELTA = 0.0001



loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_FACTOR)


selected_optimizer_class, selected_lr = select_best_optimizer_lr(
    1,
    cnn_model,
    aug_train_loader,
    loss_fn,
    GRADIENT_CLIP_VALUE,
    DEVICE
)

print(selected_optimizer_class, selected_lr)

if selected_optimizer_class is optim.SGD:
    WEIGHT_DECAY = 0.0001
    optimizer = selected_optimizer_class(
        cnn_model.parameters(), lr=selected_lr, weight_decay=WEIGHT_DECAY, momentum=0.9
    )

else:
    if selected_optimizer_class is optim.AdamW:
        WEIGHT_DECAY = 0.01
    elif selected_optimizer_class is optim.Adam:
        WEIGHT_DECAY = 0.0001
        
    optimizer = selected_optimizer_class(
        cnn_model.parameters(),
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
    cnn_model, loss_train, acc_train = train_one_epoch_m1(
        cnn_model,
        aug_train_loader,
        loss_fn,
        optimizer,
        DEVICE,
        GRADIENT_CLIP_VALUE,
        epoch=epoch + 1,
    )

    # --- Validate for one epoch ---
    current_val_loss, current_val_acc, _, _ = validation_epoch_fn(
        cnn_model, val_loader, loss_fn, DEVICE, description=f"Validation {epoch + 1}"
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
        best_model_state = copy.deepcopy(cnn_model.state_dict())
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
        break

    print("-" * 60)


if best_model_state:
    cnn_model.load_state_dict(best_model_state)
    torch.save(cnn_model, model_save_path)
    print("\n✅ Loaded best model based on validation accuracy for final testing.")
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
) = validation_epoch_fn(cnn_model, test_loader, loss_fn, DEVICE, description=f"Testing")

print(f"\n--- Test Set Results (Best Validation Model) ---")
print(f"\tTest Loss: {test_loss_final:.4f}")
print(f"\tTest Accuracy (Top-1): {test_acc_top1_final:.2f}%")
print(f"\tTest MRR: {test_mrr_final:.4f}")


k_values_for_test = [5, 10]

_, _, _, test_top_5_acc = validation_epoch_fn(
    cnn_model, test_loader, loss_fn, DEVICE, f"Testing (Top-{5})", k_for_top_k_eval=5
)
_, _, _, test_top_10_acc = validation_epoch_fn(
    cnn_model, test_loader, loss_fn, DEVICE, f"Testing (Top-{10})", k_for_top_k_eval=10
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
