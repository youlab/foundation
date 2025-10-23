import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_mask(
    input_tensor,
    mask_probability=0.15,
):
    mask = (torch.rand(input_tensor.shape, device=input_tensor.device) < mask_probability).float()
    masked_input = input_tensor * (1 - mask)

    return masked_input, mask


def evaluate(
    model,
    dataloader,
    criterion,
):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to track gradients
        for batch in dataloader:
            input_tensor = batch.float().to(device)  # Move input tensor to GPU
            masked_input, mask = apply_mask(input_tensor)  # Apply masking
            masked_input, mask = masked_input.to(device), mask.to(device)  # Ensure mask is on the same device
            _, mlm_output = model(masked_input)  # Get output

            masked_mlm_output = mlm_output[mask.bool()]
            masked_input_tensor = input_tensor[mask.bool()]

            loss = criterion(masked_mlm_output, masked_input_tensor)
            total_loss += loss.item() * input_tensor.size(0) 
    
    return total_loss / len(dataloader.dataset)  # Return average loss


class MCRTrainer:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        epochs,
        lr,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr

    def train(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()  # Suitable for regression tasks

        train_losses = []
        test_losses = []

        # Training loop
        min_test_loss = np.inf
        patience = 20
        count = 0
        for _ in trange(self.epochs):
            self.model.train()  # Set model to training mode
            total_train_loss = 0

            for batch in self.train_loader:
                # input_tensor = batch[0].float().to(device) 
                input_tensor = batch.float().to(device) 
                masked_input, mask = apply_mask(
                    input_tensor=input_tensor,
                    mask_probability=0.15,
                )  # Apply masking
                masked_input, mask = masked_input.to(device), mask.to(device)  # Ensure mask is on the same device
    
                optimizer.zero_grad()  # Zero the gradients
    
                # Forward pass
                _, mlm_output = self.model(masked_input)  # Get output
    
                loss = criterion(
                    mlm_output[mask.bool()],
                    input_tensor[mask.bool()],
                )

                total_train_loss += loss.item() * input_tensor.size(0)  # Aggregate training loss

                loss.backward()
                optimizer.step()
            scheduler.step()

            avg_train_loss = total_train_loss / len(self.train_loader.dataset)
            avg_test_loss = evaluate(
                model=self.model,
                dataloader=self.test_loader,
                criterion=criterion,
            )

            if avg_test_loss < min_test_loss:
                min_test_loss = avg_test_loss
                count = 0
            else:
                count += 1

            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            if count >= patience:
                break
        return test_losses[-1]
