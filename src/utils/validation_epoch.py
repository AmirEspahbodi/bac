import torch
import tqdm
from .average_meter import AverageMeter


def validation(model, test_loader, loss_fn, device):
    model.eval()
    loss_valid = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            loss_valid.update(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return loss_valid.avg, accuracy
