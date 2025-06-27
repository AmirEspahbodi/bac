from copy import deepcopy
import torch.optim as optim
from .train_one_epoch import train_one_epoch


def select_best_optimizer_lr(num_epochs, model, train_loader, loss_fn, device):
    best_accuracy = float("inf")
    selected_optimizer = None
    selected_lr = 1

    for lr in [0.1, 0.01, 0.001, 0.0001]:
        print(f"OPTIMIZER=SGD, LR={lr}")
        model_sgd = deepcopy(model)
        optimizer = optim.SGD(
            model_sgd.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9
        )
        for epoch in range(num_epochs):
            model_sgd, loss_train, accuracy = train_one_epoch(
                model_sgd, train_loader, loss_fn, optimizer, device, epoch=epoch
            )
            if accuracy < best_accuracy:
                selected_optimizer = optim.SGD
                selected_lr = lr
        del model_sgd

    for lr in [0.1, 0.01, 0.001, 0.0001]:
        print(f"OPTIMIZER=Adam, LR={lr}")
        model_adam = deepcopy(model)
        optimizer = optim.Adam(model_adam.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            model_adam, loss_train, accuracy = train_one_epoch(
                model_adam, train_loader, loss_fn, optimizer, epoch
            )
            if accuracy < best_accuracy:
                selected_optimizer = optim.Adam
                selected_lr = lr
        print()
        del model_adam

    for lr in [0.1, 0.01, 0.001, 0.0001]:
        print(f"OPTIMIZER=Adamw, LR={lr}")
        model_adamw = deepcopy(model)
        optimizer = optim.AdamW(model_adamw.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            model_adamw, loss_train, accuracy = train_one_epoch(
                model_adamw, train_loader, loss_fn, optimizer, epoch
            )
            if accuracy < best_accuracy:
                selected_optimizer = optim.AdamW
                selected_lr = lr
        del model_adamw

    selected_optimizer, selected_lr
