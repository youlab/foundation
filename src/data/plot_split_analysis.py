import matplotlib.pyplot as plt
import json
from pathlib import Path

from config import (
    DIR_DATA,
    DIR_DATA_PROCESSED,
)


def print_top_n_datasets(dataset_split, n=10):
    """
    Print the n largest datasets
    """
    sorted_datasets = sorted(
        dataset_split.items(),
        key=lambda x: x[1]["n_samples"],
        reverse=True
    )[:n]
    
    print(f"\nTop {n} Largest Datasets:")
    print(f"{'Rank':<6}{'Name':<50}{'Samples':<12}{'Split':<8}")
    print("-" * 76)
    for i, (filename, info) in enumerate(sorted_datasets, 1):
        print(f"{i:<6}{filename:<50}{info['n_samples']:<12}{info['split']:<8}")


def plot_sample_distribution(
    dataset_split, 
    group_by_split=False, 
    sample_range=(0, 12000),
    title=None, 
    save_path=None
):
    """
    Plot distribution of samples over datasets.
    
    Parameters
    ----------
    dataset_split : dict
        Dataset split info from config JSON
    group_by_split : bool
        If True, group by train-test split with different colors and legend
    sample_range : tuple
        (min, max) range of samples to include. Only datasets with n_samples in this range are plotted.
    title : str, optional
        Title for the plot
    save_path : str or Path, optional
        Path to save the plot
    """
    # filter by num sample range
    filtered_split = {fn: info for fn, info in dataset_split.items() if sample_range[0] <= info["n_samples"] < sample_range[1]}
    
    if group_by_split:
        # group samples by split
        train_samples = []
        test_samples = []
        
        for filename, info in filtered_split.items():
            n_samples = info["n_samples"]
            if info["split"] == "train":
                train_samples.append(n_samples)
            else:
                test_samples.append(n_samples)
        
        plt.figure(figsize=(12, 6))
        x_pos_train = list(range(len(train_samples)))
        x_pos_test = [i + len(train_samples) + 1 for i in range(len(test_samples))]
        
        plt.bar(x_pos_train, train_samples, color="blue", label="Train", alpha=0.8)
        plt.bar(x_pos_test, test_samples, color="red", label="Test", alpha=0.8)
        
        plt.legend()
    else:
        samples = [info["n_samples"] for info in filtered_split.values()]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(samples)), samples, color="gray", alpha=0.8)
    
    plt.xlabel("Dataset")
    plt.ylabel("Number of Samples")
    if title: plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    else:
        plt.show()



def main():
    """
    Plot sample distribution and train-test split percentages for all splitting runs
    """
    data_processed_dir = Path(DIR_DATA_PROCESSED)
    subdirs = [d for d in data_processed_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not subdirs:
        print("No subdirectories of data splitting runs found")
        return
    
    # store percentages for final plotting
    split_data = {
        "all": {"seeds": [], "train_pcts": [], "test_pcts": []},
        "simulation": {"seeds": [], "train_pcts": [], "test_pcts": []},
        "experimental": {"seeds": [], "train_pcts": [], "test_pcts": []},
    }
    
    for run_dir in sorted(subdirs):
        print(f"\nProcessing {run_dir.name}")

        figs_dir = run_dir / "split_dist_figs"
        figs_dir.mkdir(exist_ok=True)
        
        for category in ["all", "simulation", "experimental"]:
            config_file = run_dir / f"data_config_{category}.json"
            
            if not config_file.exists():
                print(f"Warning: {config_file} not found")
                continue
            
            print(f"Processing data_config_{category}.json")
            with open(config_file, "r") as f:
                config = json.load(f)
            dataset_split = config.get("dataset_split", {})
            
            # print top n datasets
            print(f"Total number of datasets: {len(dataset_split)}")
            print_top_n_datasets(dataset_split, n=10)
            
            # plot grouped with full range
            plot_sample_distribution(
                dataset_split,
                group_by_split=True,
                sample_range=(0, 100000),
                title=f"{category.upper()} Sample Distribution ",
                save_path=figs_dir / f"sample_dist_grouped_{category}.png",
            )
            print(f"Saved: sample_dist_grouped_{category}.png")

            # plot grouped with low sample range
            plot_sample_distribution(
                dataset_split,
                group_by_split=True,
                sample_range=(0, 6000),
                title=f"{category.upper()} Sample Distribution (0-6k)",
                save_path=figs_dir / f"sample_dist_grouped_{category}_0_6k.png",
            )
            print(f"Saved: sample_dist_grouped_{category}_0_6k.png")
            
            # plot grouped with high sample range
            plot_sample_distribution(
                dataset_split,
                group_by_split=True,
                sample_range=(6000, 100000),
                title=f"{category.upper()} Sample Distribution (6k-100k)",
                save_path=figs_dir / f"sample_dist_grouped_{category}_6k_100k.png",
            )
            print(f"Saved: sample_dist_grouped_{category}_6k_100k.png")
            
            split_data[category]["seeds"].append(config.get("random_seed"))
            split_data[category]["train_pcts"].append(config.get("train_percentage", 0))
            split_data[category]["test_pcts"].append(config.get("test_percentage", 0))
    
    # plot train-test percentages for each splitting run
    for category in ["all", "simulation", "experimental"]:
        if not split_data[category]["seeds"]:
            continue
        
        plt.figure(figsize=(12, 6))
        x_pos = range(len(split_data[category]["seeds"]))
        width = 0.35
        
        x_pos_train = [x - width/2 for x in x_pos]
        x_pos_test = [x + width/2 for x in x_pos]
        
        plt.bar(x_pos_train, split_data[category]["train_pcts"], width=width, label="Train %", alpha=0.8, color="blue")
        plt.bar(x_pos_test, split_data[category]["test_pcts"], width=width, label="Test %", alpha=0.8, color="red")
        
        plt.axhline(y=80, color="darkblue", linestyle="--", linewidth=2, label="Target Train (0.8)")
        plt.axhline(y=20, color="darkred", linestyle="--", linewidth=2, label="Target Test (0.2)")
        
        plt.xlabel("Random Seed")
        plt.ylabel("Percentage (%)")
        plt.title(f"Train-Test Split Percentages for {category.upper()} Data")
        plt.xticks(x_pos, split_data[category]["seeds"], rotation=45)
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        
        save_path = data_processed_dir / f"split_percentages_{category}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()