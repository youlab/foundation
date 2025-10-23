import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torchdiffeq import odeint
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import trange


def warmup_schedule(epoch,):
    warmup_epochs = 10
    return (epoch / warmup_epochs) ** 2


class NeuralTrainer:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        epochs=20000,
        patience=50,
        warmup_epochs=20,
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
    ):
        epochs_no_improve = 0
        best_test_loss = np.inf
        with logging_redirect_tqdm():
            for epoch in trange(self.epochs):
                self.model.train()
                running_loss = []

                for z0, mu, z, t in self.train_loader:
                    t = t[0]
                    self.optimizer.zero_grad()
                    z_pred = odeint(lambda t, y: ode_system(t, y, self.model, mu), z0, t, method='rk4',)[:, :, 0]
                    loss = self.criterion(
                        z_pred,
                        z,
                    )
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
                        for z0, mu, z, t in self.test_loader:
                            t = t[0]
                            z_pred = odeint(lambda t, y: ode_system(t, y, self.model, mu), z0, t, method='rk4',)[:, :, 0]
                            
                            loss = self.criterion(
                                z_pred,
                                z,
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
