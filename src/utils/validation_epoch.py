import torch
import tqdm
from .average_meter import AverageMeter
from .metrics import calculate_mrr_fn, top_k_accuracy_fn

# --- Evaluation Function with tqdm, Top-k, and MRR ---
def evaluate_epoch_fn(model_to_eval, data_loader, loss_criterion, current_device, description="", k_for_top_k_eval=None):
    model_to_eval.eval()
    epoch_loss = AverageMeter()
    epoch_accuracy_top1 = AverageMeter()
    all_predictions_eval = []
    all_labels_eval = []
    progress_bar = tqdm(data_loader, desc=description, leave=False, unit="batch")
    with torch.no_grad():
        for texts_padded, labels_batch in progress_bar:
            texts_padded, labels_batch = texts_padded.to(current_device), labels_batch.to(current_device)
            predictions = model_to_eval(texts_padded)
            loss = loss_criterion(predictions, labels_batch)
            acc_top1 = (predictions.argmax(dim=1) == labels_batch).float().mean().item()
            epoch_loss.update(loss.item(), labels_batch.size(0))
            epoch_accuracy_top1.update(acc_top1, labels_batch.size(0))
            all_predictions_eval.append(predictions.cpu())
            all_labels_eval.append(labels_batch.cpu())
            # Corrected line:
            progress_bar.set_postfix(loss=f"{epoch_loss.avg:.4f}", acc=f"{epoch_accuracy_top1.avg:.4f}")
    progress_bar.close()
    calculated_top_k_acc = None
    calculated_mrr = 0.0
    if all_predictions_eval:
        all_predictions_tensor = torch.cat(all_predictions_eval)
        all_labels_tensor = torch.cat(all_labels_eval)
        calculated_mrr = calculate_mrr_fn(all_predictions_tensor, all_labels_tensor)
        if k_for_top_k_eval is not None:
            calculated_top_k_acc = top_k_accuracy_fn(all_predictions_tensor, all_labels_tensor, k=k_for_top_k_eval)
    return epoch_loss.avg, epoch_accuracy_top1.avg, calculated_mrr, calculated_top_k_acc
