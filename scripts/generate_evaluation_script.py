#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""generate_evaluation_script.py

This script generates the evaluation scripts to test out all the model variants.
"""

import argparse
import os

from mhciipresentation.paths import LOGS_DIR


def find_best_model(performance_metric, features, data):
    paths = os.listdir(LOGS_DIR)
    selected_model_variant = [
        path for path in paths if features in path and data in path
    ]
    if data in ["nod", "iedb"]:
        selected_model_variant = [
            path for path in selected_model_variant if "iedb_nod" not in path
        ]
        if len(selected_model_variant) > 1:
            raise ValueError("Multiple model variants fulfill those criteria.")

    selected_model_variant = selected_model_variant[0]
    model_variant_path = os.path.join(
        LOGS_DIR, selected_model_variant + "/checkpoints"
    )
    epochs_model_variant = os.listdir(model_variant_path)

    # Find the best model containing highest epoch and contains the performance metrics
    relevant_checkpoints = [
        path for path in epochs_model_variant if performance_metric in path
    ]
    relevant_metric = sorted(
        relevant_checkpoints, key=lambda x: int(x.split("_")[2])
    )
    best_model = relevant_metric[-1]
    return LOGS_DIR / selected_model_variant / "checkpoints" / best_model


def find_best_models():
    best_models = dict()
    for data in ["iedb", "nod", "iedb_nod"]:
        for features in ["seq_only", "seq_mhc"]:
            best_models[hash(data + features)] = {
                "data": data,
                "features": features,
                "model": find_best_model(
                    FLAGS.performance_metric, features, data
                ),
            }
    return best_models


def main():
    best_models = find_best_models()
    with open("./scripts/evaluation_pipeline.sh", "w") as f:
        for script in ["cd4.py", "maria.py", "nod.py"]:
            for model in best_models.values():
                if model["features"] == "seq_only":
                    evaluation_script_params = f"python ./src/mhciipresentation/evaluation/{script} --model_wo_pseudo_path {model['model']} --resuts {LOGS_DIR}/evaluation/{model['features']}_{model['data']}"
                else:
                    evaluation_script_params = f"python ./src/mhciipresentation/evaluation/{script} --model_with_pseudo_path {model['model']} --resuts {LOGS_DIR}/evaluation/{model['features']}_{model['data']}"

                f.write(evaluation_script_params)
                # Add new line
                f.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--performance_metric",
        type=str,
        help="Performance metric used to select the best models.",
        default="precision",
    )
    FLAGS, unparsed = parser.parse_known_args()
    if unparsed is not None:
        print(f"Unparsed arguments: {unparsed}")
    main()
