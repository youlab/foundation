"""Checks for isolate separation and feature/target alignment."""
import unittest
from unittest.mock import patch

import numpy as np

from applications.utils.data import cross_val_data_split, group_cross_val_data_split
from applications.antibiotics.generate_results import split_data_and_get_results


class StrainSplitTests(unittest.TestCase):
    def test_group_folds_are_disjoint_exhaustive_and_aligned(self):
        groups = np.repeat(np.arange(11), np.arange(1, 12))
        rows = np.arange(len(groups))
        all_test = []
        for fold in range(5):
            train, test = group_cross_val_data_split(rows, groups, fold)
            self.assertFalse(set(groups[train]) & set(groups[test]))
            np.testing.assert_array_equal(np.sort(np.r_[train, test]), rows)
            features = np.c_[rows, rows * 3]
            x_train, x_test = group_cross_val_data_split(features, groups, fold)
            np.testing.assert_array_equal(x_train[:, 0], train)
            np.testing.assert_array_equal(x_test[:, 0], test)
            all_test.extend(test)
        np.testing.assert_array_equal(np.sort(all_test), rows)

    def test_invalid_groups_or_fold_fail(self):
        for groups, fold in [
            (np.arange(9), 0),
            (np.ones((10, 1)), 0),
            (np.arange(10), -1),
            (np.arange(10), 5),
        ]:
            with self.assertRaises(ValueError):
                group_cross_val_data_split(np.arange(10), groups, fold)

    def test_pipeline_preserves_alignment_and_default_split(self):
        rows = np.arange(60)
        features = rows[:, None]
        groups = np.repeat(np.arange(10), 6)
        for selected_groups in [groups, None]:
            with patch(
                "applications.antibiotics.generate_results.loop_for_results"
            ) as loop:
                split_data_and_get_results(
                    x_raw=features,
                    x_latents=features * 2,
                    x_latents_fine_tuned=None,
                    x_latents_end2end=None,
                    tgt=rows,
                    classify=True,
                    detailed_binary=True,
                    train_sizes=[0.8],
                    cross_val=2,
                    groups=selected_groups,
                )
            args = loop.call_args.kwargs
            for part in ["train_original", "test"]:
                np.testing.assert_array_equal(
                    args[f"x_raw_{part}"][:, 0], args[f"tgt_{part}"]
                )
                np.testing.assert_array_equal(
                    args[f"x_latent_{part}"][:, 0], args[f"tgt_{part}"] * 2
                )
            if selected_groups is None:
                _, expected = cross_val_data_split(rows, 2, 5)
                np.testing.assert_array_equal(args["tgt_test"], expected)
            else:
                self.assertFalse(
                    set(groups[args["tgt_train_original"]])
                    & set(groups[args["tgt_test"]])
                )


if __name__ == "__main__":
    unittest.main()
