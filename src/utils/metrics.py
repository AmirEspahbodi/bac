import torch
import numpy as np
import matplotlib as plt


# --- Plotting Function ---
def plot_training_history(
    num_total_epochs, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist
):
    """Plots the training and validation loss and accuracy."""
    if (
        not train_loss_hist
        or not val_loss_hist
        or not train_acc_hist
        or not val_acc_hist
    ):
        print("History lists are empty. Cannot plot.")
        return

    epochs_range = range(1, num_total_epochs + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_hist, "bo-", label="Training Loss")
    plt.plot(epochs_range, val_loss_hist, "ro-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc_hist, "bo-", label="Training Accuracy (Top-1)")
    plt.plot(epochs_range, val_acc_hist, "ro-", label="Validation Accuracy (Top-1)")
    plt.title("Training and Validation Accuracy (Top-1)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
