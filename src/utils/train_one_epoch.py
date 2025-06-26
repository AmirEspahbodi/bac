import torch
from tqdm import tqdm
from .average_meter import AverageMeter


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch=None):
    model.train()
    loss_train = AverageMeter()
    correct = 0
    total = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            if epoch is not None:
                tepoch.set_description(f"Epoch {epoch}")
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train.update(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            tepoch.set_postfix(loss=loss_train.avg, accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    return model, loss_train.avg, accuracy
