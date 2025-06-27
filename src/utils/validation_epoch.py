import torch
from tqdm import tqdm
from .average_meter import AverageMeter
from .metrics import calculate_mrr_fn, top_k_accuracy_fn

def validation_epoch_fn(model, data_loader, loss_fn, device, description="", k_for_top_k_eval=None):
    model.eval()
    epoch_loss = AverageMeter()

    all_predictions_eval = []
    all_labels_eval = []
    
    correct = 0
    total = 0

    progress_bar = tqdm(data_loader, desc=description, leave=False, unit="batch")
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)


            epoch_loss.update(loss.item())

            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_predictions_eval.append(outputs)
            all_labels_eval.append(targets)
            
            # Corrected line
            progress_bar.set_postfix(loss=f"{epoch_loss.avg:.4f}", acc=f"{100 * correct / total:.4f}")

    accuracy = 100 * correct / total
    
    progress_bar.close()
    calculated_top_k_acc = None
    calculated_mrr = 0.0
    if all_predictions_eval:
        all_predictions_tensor = torch.cat(all_predictions_eval)
        all_labels_tensor = torch.cat(all_labels_eval)
        calculated_mrr = calculate_mrr_fn(all_predictions_tensor, all_labels_tensor)
        if k_for_top_k_eval is not None:
            calculated_top_k_acc = top_k_accuracy_fn(all_predictions_tensor, all_labels_tensor, k=k_for_top_k_eval)
    return epoch_loss.avg, accuracy, calculated_mrr, calculated_top_k_acc
