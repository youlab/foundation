from ml.trainers.neural_trainer import NeuralTrainer
from ml.neural_system import get_data_loaders, NeuralODE

if __name__ == "__main__":
    
    train_loader, test_loader = get_data_loaders()
    
    model = NeuralODE()

    trainer = NeuralTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=10,
    )
