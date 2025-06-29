from copy import deepcopy
import torch.optim as optim
from .train_one_epoch import train_one_epoch_m1


def select_best_optimizer_lr(num_epochs, model, train_loader, loss_fn, GRADIENT_CLIP_VALUE, device):
    best_accuracy = -1
    best_loss = float("-inf")
    selected_optimizer = None
    selected_lr = 1

    for lr in [0.1, 0.01, 0.001, 0.0001]:
        print(f"OPTIMIZER=SGD, LR={lr}")
        model_sgd = deepcopy(model)
        optimizer = optim.SGD(
            model_sgd.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9
        )
        for epoch in range(num_epochs):
            model_sgd, loss_train, accuracy = train_one_epoch_m1(
                model_sgd, train_loader, loss_fn, optimizer, device, GRADIENT_CLIP_VALUE, epoch=epoch
            )
            if accuracy == best_accuracy:
                if best_loss < loss_train:
                    best_loss = loss_train
                    selected_optimizer = optim.SGD
                    selected_lr = lr
            elif accuracy > best_accuracy:
                best_accuracy = accuracy
                selected_optimizer = optim.SGD
                selected_lr = lr
        del model_sgd

    for lr in [0.1, 0.01, 0.001, 0.0001]:
        print(f"OPTIMIZER=Adam, LR={lr}")
        model_adam = deepcopy(model)
        optimizer = optim.Adam(model_adam.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            model_adam, loss_train, accuracy = train_one_epoch_m1(
                model_adam, train_loader, loss_fn, optimizer, device, GRADIENT_CLIP_VALUE, epoch=epoch
            )
            if accuracy == best_accuracy:
                if best_loss < loss_train:
                    best_loss = loss_train
                    selected_optimizer = optim.Adam
                    selected_lr = lr
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                selected_optimizer = optim.Adam
                selected_lr = lr
        del model_adam

    for lr in [0.1, 0.01, 0.001, 0.0001]:
        print(f"OPTIMIZER=Adamw, LR={lr}")
        model_adamw = deepcopy(model)
        optimizer = optim.AdamW(model_adamw.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            model_adamw, loss_train, accuracy = train_one_epoch_m1(
                model_adamw, train_loader, loss_fn, optimizer, device, GRADIENT_CLIP_VALUE, epoch=epoch
            )
            if accuracy == best_accuracy:
                if best_loss < loss_train:
                    best_loss = loss_train
                    selected_optimizer = optim.AdamW
                    selected_lr = lr
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                selected_optimizer = optim.AdamW
                selected_lr = lr
        del model_adamw

    return selected_optimizer, selected_lr
