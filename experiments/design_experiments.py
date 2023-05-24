"""design_experiments.py

This is simply a utility to design all of the experiments that we want to run.
"""

import itertools

import pandas as pd

pd.set_option("display.max_rows", 500)


def main():
    feature_set = ["seq_only", "seq_mhc"]
    data_source = ["iedb", "iedb_nod", "nod"]
    layers = [2, 4, 8]
    seeds = range(3)
    # Create all combinations using itertools.product
    combinations = list(
        itertools.product(feature_set, data_source, layers, seeds)
    )
    experiments = pd.DataFrame(
        combinations, columns=["feature_set", "data_source", "layers", "seed"]
    )
    experiments["exp_type"] = "train_variants"
    experiments["pos"] = "pos"
    for seed in seeds:
        experiments = pd.concat(
            [
                experiments,
                pd.DataFrame(
                    {
                        "feature_set": "best_variant",
                        "data_source": "best_variant",
                        "layers": "best_variant",
                        "seed": seed,
                        "exp_type": "train_ablation",
                        "pos": "no_pos",
                    },
                    index=[0],
                ),
            ],
        )

    # Inference of model on evaluation datasets
    inference_datasets = [
        "strattmann",
        "you",
        "xu",
        "K562 DRB1*04:04",
        "K562 DRB1*01:01",
        "Melanoma",
    ]
    for seed in seeds:
        for inf in inference_datasets:
            experiments = pd.concat(
                [
                    experiments,
                    pd.DataFrame(
                        {
                            "feature_set": "best_variant",
                            "data_source": "best_variant",
                            "layers": "best_variant",
                            "seed": seed,
                            "exp_type": f"inference_{inf}",
                            "pos": "pos",
                        },
                        index=[0],
                    ),
                ],
            )
    experiments["split"] = "ours_random"

    for seed in seeds:
        experiments = pd.concat(
            [
                experiments,
                pd.DataFrame(
                    {
                        "feature_set": "best_variant",
                        "data_source": "best_variant",
                        "layers": "best_variant",
                        "seed": seed,
                        "exp_type": "train_alternative_split",
                        "pos": "pos",
                        "split": "MHCAttnNet_random",
                    },
                    index=[0],
                ),
            ],
        )

    for seed in seeds:
        experiments = pd.concat(
            [
                experiments,
                pd.DataFrame(
                    {
                        "feature_set": "best_variant",
                        "data_source": "best_variant",
                        "layers": "best_variant",
                        "seed": seed,
                        "exp_type": "train_alternative_split",
                        "pos": "pos",
                        "split": "leave_one_allele_out",
                    },
                    index=[0],
                ),
            ],
        )

    experiments.to_csv("experiments.csv", index=False)
    experiments.to_latex("experiments.tex", index=False)


if __name__ == "__main__":
    main()
