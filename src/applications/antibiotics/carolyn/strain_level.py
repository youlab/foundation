"""Reproduce strain-held-out resistance prediction and class balance."""
import argparse
import json
import logging
from pathlib import Path
import platform
from importlib.metadata import version

import numpy as np
import sklearn

from applications.antibiotics.carolyn.data import get_data
from applications.antibiotics.config import TRAIN_SIZES
from applications.antibiotics.generate_results import split_data_and_get_results
from applications.utils.data import group_cross_val_data_split, interpolate_y

ANTIBIOTICS = ["SAM", "GM", "SXT", "CIP"]
EXPECTED_RESISTANT = dict(zip(ANTIBIOTICS, [158, 50, 155, 163]))
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[4] / "results" / "antibiotics" / "strain_level"
)
logger = logging.getLogger(__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=4, allow_nan=False) + "\n")


def verify(df, x_raw):
    groups = df["strain"].to_numpy()
    if df["strain"].isna().any() or len(groups) != len(x_raw):
        raise ValueError("Missing strain IDs or misaligned curves")
    report = dict(
        n_rows=len(df),
        n_unique_strains=len(np.unique(groups)),
        x_raw_shape=list(x_raw.shape),
        alignment_checks={"groups_match_x_raw": True},
        label_consistency_by_strain={},
        group_split_checks={},
    )
    for antibiotic in ANTIBIOTICS:
        tgt = df[antibiotic].to_numpy()
        if not np.isin(tgt, [0, 1]).all():
            raise ValueError(f"{antibiotic}: missing or non-binary labels")
        inconsistent = df.groupby("strain")[antibiotic].nunique()
        inconsistent = inconsistent[inconsistent > 1]
        per_strain = df.groupby("strain")[antibiotic].max()
        report["alignment_checks"][f"{antibiotic}_target_matches_groups"] = len(
            tgt
        ) == len(groups)
        report["label_consistency_by_strain"][antibiotic] = dict(
            n_inconsistent_strains=len(inconsistent),
            inconsistent_strains=list(map(str, inconsistent.index)),
            total_strains=len(per_strain),
            resistant_strains=int(per_strain.sum()),
            susceptible_strains=int((per_strain == 0).sum()),
            resistant_fraction=float(per_strain.mean()),
            matches_reported_balance=bool(
                len(per_strain) == 277
                and per_strain.sum() == EXPECTED_RESISTANT[antibiotic]
            ),
        )
        if len(inconsistent):
            raise ValueError(f"{antibiotic}: inconsistent labels within strains")
        checks = report["group_split_checks"][antibiotic] = {}
        test_indices = []
        for fold in range(5):
            train, test = group_cross_val_data_split(np.arange(len(df)), groups, fold)
            train_strains, test_strains = set(groups[train]), set(groups[test])
            overlap = train_strains & test_strains
            if overlap:
                raise ValueError(f"{antibiotic} fold {fold}: overlapping strains")
            test_indices.extend(test.tolist())
            checks[str(fold)] = dict(
                train_samples=len(train),
                test_samples=len(test),
                train_strains=len(train_strains),
                test_strains=len(test_strains),
                overlapping_strains=len(overlap),
                train_strain_ids=sorted(map(str, train_strains)),
                test_strain_ids=sorted(map(str, test_strains)),
                train_positive_samples=int(tgt[train].sum()),
                test_positive_samples=int(tgt[test].sum()),
                train_positive_fraction=float(tgt[train].mean()),
                test_positive_fraction=float(tgt[test].mean()),
            )
        if sorted(test_indices) != list(range(len(df))):
            raise ValueError("Each curve must appear in exactly one test fold")
    return report


def get_final_metric(results, model_key, metric_key="f1_score"):
    final_size = str(max(int(k) for k in results if k != "test_size"))
    return results[final_size][model_key][metric_key]


def summarize(output_dir, report, model_type, z_dim):
    summary = dict(
        model_type=model_type,
        z_dim=z_dim,
        metric="f1_score",
        split="strain_level_group_kfold",
        std_ddof=0,
        evaluation_unit="curve",
        train_sizes=TRAIN_SIZES,
        classifier=dict(
            type="ExtraTreesClassifier",
            random_state=42,
            n_estimators=100,
            max_depth=None,
        ),
        pretraining_note="Unlabeled clinical-isolate curves were included in A7X pretraining; isolates are held out only from supervised classifier fitting.",
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            sklearn=sklearn.__version__,
            torch=version("torch"),
            huggingface_hub=version("huggingface-hub"),
        ),
        antibiotics={},
    )
    metadata_path = output_dir / "model_provenance.json"
    if metadata_path.exists():
        summary["model_provenance"] = json.loads(metadata_path.read_text())
    for antibiotic in ANTIBIOTICS:
        scores = {"raw": [], "latent": []}
        for fold in range(5):
            results = json.loads(
                (
                    output_dir
                    / f"classify_antibiotics_{antibiotic}_{fold}_{model_type}_{z_dim}.json"
                ).read_text()
            )
            for name in scores:
                scores[name].append(get_final_metric(results, f"{name}_accuracy"))
        performance = {}
        for name, values in scores.items():
            performance.update(
                {
                    f"{name}_scores": values,
                    f"{name}_mean": float(np.mean(values)),
                    f"{name}_std": float(np.std(values)),
                }
            )
        summary["antibiotics"][antibiotic] = dict(
            performance=performance,
            class_balance_by_strain=report["label_consistency_by_strain"][antibiotic],
        )
    path = output_dir / f"strain_split_summary_with_balance_{model_type}_{z_dim}.json"
    write_json(path, summary)
    return summary


def plot(summary, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    positions = np.arange(4)
    for name, offset, color, label in [
        ("raw", -0.08, "tab:blue", "Raw curve"),
        ("latent", 0.08, "tab:orange", f'{summary["model_type"]} latent'),
    ]:
        perf = [summary["antibiotics"][ab]["performance"] for ab in ANTIBIOTICS]
        axes[0].errorbar(
            positions + offset,
            [p[f"{name}_mean"] for p in perf],
            yerr=[p[f"{name}_std"] for p in perf],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
    axes[0].set(
        xticks=positions,
        xticklabels=ANTIBIOTICS,
        ylim=(0, 1.05),
        ylabel="F1 score",
        title="(a) Strain-held-out prediction",
    )
    axes[0].legend()
    balances = [
        summary["antibiotics"][ab]["class_balance_by_strain"] for ab in ANTIBIOTICS
    ]
    axes[1].bar(ANTIBIOTICS, [b["resistant_fraction"] for b in balances], color="0.5")
    for i, b in enumerate(balances):
        axes[1].text(
            i,
            b["resistant_fraction"] + 0.025,
            f"{b['resistant_strains']}/{b['total_strains']}",
            ha="center",
        )
    axes[1].set(
        ylim=(0, 1), ylabel="Resistant isolate fraction", title="(b) Class balance"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(
            output_dir
            / f'strain_level_{summary["model_type"]}_{summary["z_dim"]}.{ext}',
            dpi=200,
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", default="A7X")
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--latent-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.latent_batch_size < 1:
        parser.error("--latent-batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = get_data()
    x_raw = interpolate_y(df.loc[:, 0:98].to_numpy())
    report = verify(df, x_raw)
    write_json(args.output_dir / "strain_split_data_verification.json", report)
    for antibiotic, balance in report["label_consistency_by_strain"].items():
        logger.info("%s: %s", antibiotic, balance)
    if args.verify_only:
        return
    from applications.utils.latents import get_latents
    from ml.utils.load_models import load_model

    model = load_model(args.model_type, args.z_dim)
    if args.model_type == "A7X":
        from huggingface_hub import try_to_load_from_cache
        from ml.utils.formatting import num_to_str

        repo_id = f"you-lab/autoencoder7x-{num_to_str(args.z_dim)}"
        config_path = try_to_load_from_cache(repo_id, "config.json")
        write_json(
            args.output_dir / "model_provenance.json",
            dict(
                repo_id=repo_id,
                cached_revision=Path(config_path).parent.name
                if isinstance(config_path, str)
                else None,
                latent_batch_size=args.latent_batch_size,
            ),
        )
    logger.info(
        "Encoding %d curves using latent batches of %d",
        len(x_raw),
        args.latent_batch_size,
    )
    x_latents = get_latents(
        x_raw,
        model=model,
        z_dim=args.z_dim,
        batch_size=args.latent_batch_size,
        use_transformer=args.model_type == "MCR",
    )
    for fold in range(5):
        for antibiotic in ANTIBIOTICS:
            logger.info("Running fold %d, antibiotic %s", fold, antibiotic)
            results = split_data_and_get_results(
                x_raw=x_raw,
                x_latents=x_latents,
                x_latents_fine_tuned=None,
                x_latents_end2end=None,
                tgt=df[antibiotic].to_numpy(),
                classify=True,
                detailed_binary=True,
                train_sizes=TRAIN_SIZES,
                cross_val=fold,
                groups=df["strain"].to_numpy(),
            )
            write_json(
                args.output_dir
                / f"classify_antibiotics_{antibiotic}_{fold}_{args.model_type}_{args.z_dim}.json",
                results,
            )
    summary = summarize(args.output_dir, report, args.model_type, args.z_dim)
    plot(summary, args.output_dir)
    for antibiotic, result in summary["antibiotics"].items():
        logger.info("%s: %s", antibiotic, result["performance"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
