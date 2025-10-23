import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import trange


def warmup_schedule(epoch,):
    warmup_epochs = 10
    return (epoch / warmup_epochs) ** 2


def apply_mask_fcn(input_tensor, mask_probability=0.15):
    mask = (torch.rand(input_tensor.shape, device=input_tensor.device) < mask_probability).float()
    masked_input = input_tensor * (1 - mask)

    return masked_input, mask


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        epochs=20000,
        patience=50,
        warmup_epochs=20,
        alpha=1e-4,
        weight_decay=1e-5,
        lr=2e-4,
        min_lr_mult=1e-5,
        gamma=0.996,
        test_frequency=10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.test_frequency = test_frequency
        self.alpha = alpha
        self.patience = patience

        self.train_loss = []
        self.test_loss = []
        
        self.min_lr = lr * min_lr_mult

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler_warmup = LambdaLR(
            self.optimizer,
            lr_lambda=warmup_schedule,
        )

        self.scheduler_exponential = ExponentialLR(
            self.optimizer,
            gamma=gamma,
        )
    
    def train(
        self,
        apply_mask=False
    ):
        epochs_no_improve = 0
        best_test_loss = np.inf
        with logging_redirect_tqdm():
            for epoch in trange(self.epochs):
                self.model.train()
                running_loss = []

                for x in self.train_loader:
                    self.optimizer.zero_grad()
                    if apply_mask:
                        x, mask = apply_mask_fcn(
                            input_tensor=x,
                        )
                    recon, mu, logvar = self.model(x)
                    if torch.isnan(recon).any():
                        np.save("x.npy", x.cpu().detach().numpy(),)
                        raise ValueError("Reconstruction has NaN.")
                            
                    if apply_mask:
                        recon_loss = self.criterion(
                            recon[mask.bool()],
                            x[mask.bool()],
                        )
                    else:
                        recon_loss = self.criterion(
                            recon[x!=-1],
                            x[x!=-1],
                        )
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + self.alpha * kl_loss
                    loss.backward()
                    self.optimizer.step()
                    running_loss.append(loss.item() / x.size(0))
                
                if epoch > self.warmup_epochs:
                    self.scheduler_exponential.step()
                    self.optimizer.param_groups[0]['lr'] = max(
                        self.optimizer.param_groups[0]['lr'],
                        self.min_lr,
                    )
                else:
                    self.scheduler_warmup.step()

                self.train_loss.append(np.mean(running_loss))

                if epoch % self.test_frequency == 0:
                    self.model.eval()
                    running_loss = []
                    with torch.no_grad():
                        for x in self.test_loader:
                            recon, _, _ = self.model(x)
                            if torch.isnan(recon).any():
                                raise ValueError("Reconstruction has NaN.")
                            loss = self.criterion(
                                recon[x!=-1],
                                x[x!=-1],
                            )
                            running_loss.append(loss.item())
                    self.test_loss.append(np.mean(running_loss))

                    if self.test_loss[-1] < best_test_loss:
                        best_test_loss  = self.test_loss[-1]
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    
                    if epochs_no_improve == self.patience:
                        break
                
                else:
                    self.test_loss.append(self.test_loss[-1])
                    epochs_no_improve += 1
        return self.test_loss[-1]
    def eval(
        self,
        loader,
    ):
        recons = []
        mus = []
        logvars = []
        running_loss = []
        with torch.no_grad():
            for x in loader:
                recon, mu, logvar = self.model(
                    x,
                    train=False,
                )
                loss = self.criterion(
                    recon[x!=-1],
                    x[x!=-1],
                )
                running_loss.append(loss.item())
                recons.append(recon)
                mus.append(mu)
                logvars.append(logvar)
        return recons, mus, logvars, np.mean(running_loss)
