import torch
from utils import DEVICE
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, accuracy_fn, device=DEVICE):
        self.model = model
        self.accuracy_fn = accuracy_fn
        self.device = device

    # Setup Loss and Optimizer
    def setup(self, num_classes: int, learning_rate: float, momentum: float):
        if num_classes == 2:
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum
        )

    def train_step(self, data_loader):
        train_loss, train_acc = 0, 0
        self.model.to(self.device)
        for batch, (X, y) in enumerate(data_loader):
            # Send data to GPU
            X, y = X.to(self.device), y.to(self.device)

            # 1. Forward pass
            y_pred = self.model(X)

            # 2. Calculate loss
            loss = self.loss_fn(y_pred, y)
            train_loss += loss
            train_acc += self.accuracy_fn(
                y_true=y, y_pred=y_pred.argmax(dim=1)
            )  # Go from logits -> pred labels

            # 3. Optimizer zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()

        # Calculate loss and accuracy per epoch and print out what's happening
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    def test_step(self, data_loader):
        test_loss, test_acc = 0, 0
        self.model.to(self.device)
        self.model.eval()  # put model in eval mode
        # Turn on inference context manager
        with torch.inference_mode():
            for X, y in data_loader:
                # Send data to GPU
                X, y = X.to(self.device), y.to(self.device)

                # 1. Forward pass
                test_pred = self.model(X)

                # 2. Calculate loss and accuracy
                test_loss += self.loss_fn(test_pred, y)
                test_acc += self.accuracy_fn(
                    y_true=y,
                    y_pred=test_pred.argmax(dim=1),  # Go from logits -> pred labels
                )

            # Adjust metrics and print out
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
