import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import (
    DIR_RESULTS_MODEL,
    DIR_RESULTS_MODEL_NEW_SPLITS,
)


def plot_loss_curve(
    train_loss_file=None, 
    test_loss_file=None, 
    n_epochs=None, 
    title=None, 
    save_path=None
):
    """
    Plot loss curves stored in .npy files
    """
    plt.figure(figsize=(10, 6))
    
    if train_loss_file:
        train_loss = np.load(train_loss_file) * 2048
        if n_epochs:
            train_loss = train_loss[:n_epochs]
        plt.plot(train_loss, label="Train Loss", color="blue")
    
    if test_loss_file:
        test_loss = np.load(test_loss_file)
        if n_epochs:
            test_loss = test_loss[:n_epochs]
        plt.plot(test_loss, label="Test Loss", color="red")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title or "Loss Curves")
    plt.yscale("log")
    plt.grid(True, alpha=0.5)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    else:
        plt.show()


def main(n_epochs=None):
    # n_epochs optional, stores number of epochs to plot
    
    model_dirs = [DIR_RESULTS_MODEL, DIR_RESULTS_MODEL_NEW_SPLITS]

    for model_dir in model_dirs:

        model_dir = Path(model_dir)
        if not model_dir.exists():
            print(f"Directory does not exist: {model_dir}")
            continue
        print(f"\nProcessing models in {model_dir}")
    
        # find subdirs where model info is saved
        subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not subdirs:
            print("No subdirectories found")
            continue
        
        for subdir in subdirs:
            print(f"Processing {subdir.name}")
            
            train_loss_file = subdir / "train_loss.npy"
            test_loss_file = subdir / "test_loss.npy"
            
            if not train_loss_file.exists() and not test_loss_file.exists():
                print(f"Warning: No loss files found in {subdir}")
                continue
            
            if n_epochs: 
                loss_curves_png = subdir / f"loss_curves_{n_epochs}.png"
            else: 
                loss_curves_png = subdir / "loss_curves.png"
            
            plot_loss_curve(
                train_loss_file=train_loss_file if train_loss_file.exists() else None,
                test_loss_file=test_loss_file if test_loss_file.exists() else None,
                n_epochs=n_epochs,
                title=f"{subdir.name} Loss Curves",
                save_path=loss_curves_png,
            )
            
            print(f"Saved loss curve at {loss_curves_png}")
