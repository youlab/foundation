class PRTrainer:

    def __init__(
        self,
        model,
        train_loader,
    ):
        self.model = model
        self.train_loader = train_loader

    def train(self):
        for x in self.train_loader:
            x = x.cpu().detach().numpy()[:, 0, :]
            self.model.partial_fit(x)
