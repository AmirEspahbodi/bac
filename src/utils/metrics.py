import torch
import numpy as np



# --- Top-k Accuracy Function ---
def top_k_accuracy_fn(outputs, targets, k=3):
    batch_size = targets.size(0)
    if batch_size == 0:
        return 0.0
    if k > outputs.size(1):  # If k is larger than num_classes
        print(
            f"Warning: k ({k}) for top-k accuracy is greater than the number of classes ({outputs.size(1)}). Calculating for k={outputs.size(1)}."
        )
        k = outputs.size(1)
    if k == 0:
        return 0.0  # Should not happen if k >=1

    _, pred_indices = outputs.topk(k, dim=1, largest=True, sorted=True)
    targets_expanded = targets.view(-1, 1).expand_as(pred_indices)
    correct = pred_indices.eq(targets_expanded).sum().item()
    return correct / batch_size


# --- MRR Calculation Function ---
def calculate_mrr_fn(outputs, targets):
    if outputs.size(0) == 0:
        return 0.0

    # Get sorted indices of predictions (highest score first)
    sorted_indices = torch.argsort(outputs, dim=1, descending=True)

    reciprocal_ranks = []
    for i in range(targets.size(0)):
        target_label = targets[i]
        try:
            rank = (sorted_indices[i] == target_label).nonzero(as_tuple=True)[
                0
            ].item() + 1
            reciprocal_ranks.append(1.0 / rank)
        except IndexError:
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0
    return np.mean(reciprocal_ranks)

