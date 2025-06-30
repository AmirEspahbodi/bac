import torch
from tqdm import tqdm
from .average_meter import AverageMeter


def train_one_epoch_m1(
    model, train_loader, loss_fn, optimizer, device, clip_value, epoch=None
):
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
            
            print("\nDEBUG: - - - - -")
            print(f"inputs.shape = {inputs.shape}")
            print(f"targets.shape = {targets.shape}")
            # Forward pass
            logits = model(inputs)

            print(f"logits.shape = {logits.shape}")
            print("\nDEBUG: - - - - -")
            
            loss = loss_fn(logits, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # ** Gradient Clipping **
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            loss_train.update(loss.item())

            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            tepoch.set_postfix(loss=loss_train.avg, accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    return model, loss_train.avg, accuracy
