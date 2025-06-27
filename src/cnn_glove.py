import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import get_data_loaders
from .models import CNNModel
from .utils import (
    plot_training_history,
    train_one_epoch,
    evaluate_epoch_fn,
    select_best_optimizer_lr,
)
import copy
import json

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


selected_optimizer_class, selected_lr = select_best_optimizer_lr(
    1,
    cnn_model,
    aug_train_loader,
    loss_fn,
    DEVICE
)

print(selected_optimizer_class, selected_lr)

if selected_optimizer_class is optim.SGD:
    optimizer = selected_optimizer_class(cnn_model.parameters(), lr=selected_lr, weight_decay=WEIGHT_DECAY, momentum=0.9)
else:
    optimizer = selected_optimizer_class(cnn_model.parameters(), lr=selected_lr, weight_decay=WEIGHT_DECAY, )
    

loss_train_hist = []
loss_valid_hist = []
acc_train_hist = []
acc_valid_hist = []


best_loss_valid = torch.inf
epoch_counter = 0
best_model_state = None


for epoch in range(NUM_EPOCHS):
    # Train
    cnn_model, loss_train, acc_train = train_one_epoch(
        cnn_model, aug_train_loader, loss_fn, optimizer, epoch
    )
    # Validation
    loss_valid, acc_valid, _, _= evaluate_epoch_fn(cnn_model, test_loader, loss_fn)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)

    acc_train_hist.append(acc_train)
    acc_valid_hist.append(acc_valid)

    if loss_valid < best_loss_valid:
        best_loss_valid = loss_valid
        best_model_state = copy.deepcopy(cnn_model.state_dict())
        print(f"\t✨ New best validation loss: {best_loss_valid:.4f}%. Model saved.")


    print(f"Valid: Loss = {loss_valid:.4}, Acc = {acc_valid:.4}")
    print()

    epoch_counter += 1
    

if best_model_state:
    cnn_model.load_state_dict(best_model_state)
    torch.save(cnn_model, f"./.models/cnn_glove_model.pt")
    print("\n✅ Loaded best model based on validation accuracy for final testing.")
else:
    print("\n⚠️ No improvement observed. Using model from the last epoch for testing.")


plot_training_history(NUM_EPOCHS, loss_train_hist, loss_valid_hist, acc_train_hist, acc_valid_hist)



print("\n🧪 Evaluating on Test Set...")
test_loss_final, test_acc_top1_final, test_mrr_final, _ = evaluate_epoch_fn(cnn_model, test_loader, optimizer, DEVICE, "Testing")

print(f"\n--- Test Set Results (Best Validation Model) ---")
print(f"\tTest Loss: {test_loss_final:.4f}")
print(f"\tTest Accuracy (Top-1): {test_acc_top1_final*100:.2f}%")
print(f"\tTest MRR: {test_mrr_final:.4f}")


k_values_for_test = [5, 10]

_, _, _, test_top_5_acc = evaluate_epoch_fn(cnn_model, test_loader, optimizer, DEVICE, f"Testing (Top-{5})", k_for_top_k_eval=5)
_, _, _, test_top_10_acc = evaluate_epoch_fn(cnn_model, test_loader, optimizer, DEVICE, f"Testing (Top-{10})", k_for_top_k_eval=10)

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
with open("./.result/cnn_glove_result.json", "w") as f:
    json.dump(result, f, indent=4)
