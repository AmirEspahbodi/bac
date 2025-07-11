import os
import argparse
import sys
import torch
import copy
import json
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
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
        choices=["glove_mean", "bert_cls", "bert_mean"],
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

match selected_embedding:
    case "glove_mean":
        aug_train_loader = single_vector_glove_dataloader(aug_train_loader)
        val_loader = single_vector_glove_dataloader(val_loader)
        test_loader = single_vector_glove_dataloader(test_loader)
        BERT_DIM = 300
        HIDDEN_DIM = 128
        NUM_BLOCKS = 4
        DROPOUT = 0.2
        model_save_path = f"{os.getcwd()}/.models/lMLPglove_model.pt"
        result_save_path = f"{os.getcwd()}/.result/lMLPglove_result.json"
    case "bert_cls:
        BERT_DIM = 768
        HIDDEN_DIM = 512
        NUM_BLOCKS = 4
        DROPOUT = 0.2
        model_save_path = f"{os.getcwd()}/.models/MLP_glove_model.pt"
        result_save_path = f"{os.getcwd()}/.result/MLP_glove_result.json"
    case "bert_mean":
        aug_train_loader = single_vector_glove_dataloader(aug_train_loader)
        val_loader = single_vector_glove_dataloader(val_loader)
        test_loader = single_vector_glove_dataloader(test_loader)
        BERT_DIM = 768
        HIDDEN_DIM = 512
        NUM_BLOCKS = 4
        DROPOUT = 0.2
        model_save_path = f"{os.getcwd()}/.models/MLP_glove_model.pt"
        result_save_path = f"{os.getcwd()}/.result/MLP_glove_result.json"

BATCH_SIZE = 32
LABEL_SMOOTHING = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
).to(DEVICE)

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

# --- Early Stopping Configuration ---
PATIENCE = 5
MIN_DELTA = 0.0001

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
    mlp_model, loss_train, acc_train = train_one_epoch_m1(
        mlp_model,
        aug_train_loader,
        loss_fn,
        optimizer,
        DEVICE,
        GRADIENT_CLIP_VALUE,
        epoch=epoch + 1,
    )

    # --- Validate for one epoch ---
    current_val_loss, current_val_acc, _, _ = validation_epoch_fn(
        mlp_model, val_loader, loss_fn, DEVICE, description=f"Validation {epoch + 1}"
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
        best_model_state = copy.deepcopy(mlp_model.state_dict())
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
    mlp_model.load_state_dict(best_model_state)
    torch.save(mlp_model, model_save_path)
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
) = validation_epoch_fn(mlp_model, test_loader, loss_fn, DEVICE, description=f"Testing")

print(f"\n--- Test Set Results (Best Validation Model) ---")
print(f"\tTest Loss: {test_loss_final:.4f}")
print(f"\tTest Accuracy (Top-1): {test_acc_top1_final:.2f}%")
print(f"\tTest MRR: {test_mrr_final:.4f}")


k_values_for_test = [5, 10]

_, _, _, test_top_5_acc = validation_epoch_fn(
    mlp_model, test_loader, loss_fn, DEVICE, f"Testing (Top-{5})", k_for_top_k_eval=5
)
_, _, _, test_top_10_acc = validation_epoch_fn(
    mlp_model, test_loader, loss_fn, DEVICE, f"Testing (Top-{10})", k_for_top_k_eval=10
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
