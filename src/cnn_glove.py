import os
import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import get_data_loaders
from .models import CNNModel
from .utils import (
    plot_training_history,
    train_one_epoch,
    validation_epoch_fn,
    select_best_optimizer_lr,
)
import copy
import json
from pathlib import Path

# --- Model & Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01 # for adamw
NUM_EPOCHS = 7
EMBEDDING_DIM_VALUE = 300
N_FILTERS_LIST = [128, 128, 128]
FILTER_SIZES_LIST = [3, 4, 5]
DROPOUT_RATE_VALUE = 0.5
HIDDEN_DIM_FC_VALUE = 256
LABEL_SMOOTHING_FACTOR = 0.1
GRADIENT_CLIP_VALUE = 1.0

# --- Early Stopping Configuration ---
PATIENCE = 5
MIN_DELTA = 0.0001


model_save_path = f"{os.getcwd()}/.models/cnn_glove_model.pt"
result_save_path=f"{os.getcwd()}/.result/cnn_glove_result.json"
Path(model_save_path).mkdir(parents=True, exist_ok=True)
Path(result_save_path).mkdir(parents=True, exist_ok=True)

aug_train_loader, val_loader, test_loader, NUM_ACTUAL_CLS = get_data_loaders()
loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_FACTOR)


cnn_model = CNNModel(
    embed_dim=EMBEDDING_DIM_VALUE,
    filter_sizes=FILTER_SIZES_LIST,
    num_filters_per_size=N_FILTERS_LIST,
    num_classes=NUM_ACTUAL_CLS,
    dropout_rate=DROPOUT_RATE_VALUE,
    hidden_dim_fc=HIDDEN_DIM_FC_VALUE,
).to(DEVICE)


selected_optimizer_class, selected_lr = optim.AdamW, 0.001 #select_best_optimizer_lr(
#     1,
#     cnn_model,
#     aug_train_loader,
#     loss_fn,
#     DEVICE
# )

print(selected_optimizer_class, selected_lr)

if selected_optimizer_class is optim.SGD:
    optimizer = selected_optimizer_class(cnn_model.parameters(), lr=selected_lr, weight_decay=WEIGHT_DECAY, momentum=0.9)
else:
    optimizer = selected_optimizer_class(cnn_model.parameters(), lr=selected_lr, weight_decay=WEIGHT_DECAY, )
    
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)

loss_train_hist = []
loss_valid_hist = []
acc_train_hist = []
acc_valid_hist = []


best_val_loss = float('inf')
epochs_no_improve = 0
early_stop_triggered = False

best_model_state = None 


print("\n✅ Start Training ...")

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    # --- Train for one epoch ---
    cnn_model, loss_train, acc_train = train_one_epoch(
        cnn_model, aug_train_loader, loss_fn, optimizer, DEVICE, GRADIENT_CLIP_VALUE, epoch=epoch+1
    )
    
    # --- Validate for one epoch ---
    current_val_loss, current_val_acc, _, _ = validation_epoch_fn(
        cnn_model, val_loader, loss_fn, DEVICE, description=f"Validation {epoch + 1}"
    )

    # --- Update history ---
    loss_train_hist = [].append(loss_train)
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
        print(f"\tValidation loss did not improve. Patience: {epochs_no_improve}/{PATIENCE}")

    # --- Step the scheduler ---
    scheduler.step()

    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}.")
        break
    
    print("-" * 60) 


if best_model_state:
    cnn_model.load_state_dict(best_model_state)
    with open(model_save_path, "w") as fp:
        torch.save(cnn_model, fp)
    print("\n✅ Loaded best model based on validation accuracy for final testing.")
else:
    print("\n⚠️ No improvement observed. Using model from the last epoch for testing.")


plot_training_history(NUM_EPOCHS, loss_train_hist, loss_valid_hist, acc_train_hist, acc_valid_hist)



print("\n🧪 Evaluating on Test Set...")
test_loss_final, test_acc_top1_final, test_mrr_final, _ = validation_epoch_fn(cnn_model, test_loader, optimizer, DEVICE, "Testing")

print(f"\n--- Test Set Results (Best Validation Model) ---")
print(f"\tTest Loss: {test_loss_final:.4f}")
print(f"\tTest Accuracy (Top-1): {test_acc_top1_final*100:.2f}%")
print(f"\tTest MRR: {test_mrr_final:.4f}")


k_values_for_test = [5, 10]

_, _, _, test_top_5_acc = validation_epoch_fn(cnn_model, test_loader, optimizer, DEVICE, f"Testing (Top-{5})", k_for_top_k_eval=5)
_, _, _, test_top_10_acc = validation_epoch_fn(cnn_model, test_loader, optimizer, DEVICE, f"Testing (Top-{10})", k_for_top_k_eval=10)

result = {
    "loss_train_hist": loss_train_hist,
    "loss_valid_hist": loss_valid_hist,
    "acc_train_hist": acc_train_hist,
    "acc_valid_hist": acc_valid_hist,
    "test_top_5_acc": test_top_5_acc,
    "test_top_10_acc": test_top_10_acc,
    "test_loss_final": test_loss_final,
    "test_acc_top1_final": test_acc_top1_final,
    "test_mrr_final": test_mrr_final
}
print(result)

# Save to JSON file
with open(result_save_path, "w") as f:
    json.dump(result, f, indent=4)
