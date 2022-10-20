#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""generate_evaluation_script.py

This script generates the evaluation scripts to test out all the model variants.
"""

import argparse
import os
import glob
import json
import re
import pandas

from mhciipresentation.paths import LOGS_DIR


def find_best_model(performance_metric, features, data):
    print("feature: %s, data: %s" % (features, data))
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
    model_variant_metrics = os.path.join(
        LOGS_DIR, selected_model_variant + "/metrics"
    )
    model_variant_checkpoints = os.path.join(
        LOGS_DIR, selected_model_variant + "/checkpoints"
    )
    checkpoint_files = glob.glob(os.path.join(model_variant_checkpoints, "*.pth"))
    metric_files=glob.glob(os.path.join(model_variant_metrics,"*.json"))

    records=list()
    for metric_json_fn in metric_files:
        record_dict=dict()
        epoch=int(re.search("epoch_(?P<epoch>[0-9]{1,3})\.json",os.path.basename(metric_json_fn)).groupdict()["epoch"])
        record_dict.update({"epoch":epoch})
        with open(metric_json_fn, "r") as metric_json:
            metrics=json.load(metric_json)
            for key in metrics:
                record_dict.update([("_".join([key,k]),v) for k,v in metrics[key].items()])
            records.append(record_dict)
    metric_df = pandas.DataFrame.from_records(records)
    metric_sorted = metric_df.sort_values(by="epoch").set_index("epoch")
    saved_epochs=sorted([int(re.search("epoch_(?P<epoch>[0-9]{1,3})_",el).groupdict()["epoch"]) for el in checkpoint_files])
    last_saved_epochs=saved_epochs[int(len(saved_epochs)-len(saved_epochs)*0.2):]
    metric_avail= metric_sorted["val_" + performance_metric].loc[last_saved_epochs]
    max_epoch = metric_avail.idxmax()
    print("max epoch: %s" % max_epoch)

    model_path = [el for el in checkpoint_files if "epoch_%s" % max_epoch in el][0]
    
    return model_path


def find_best_models():
    best_models = dict()
    #for data in ["iedb", "nod", "iedb_nod"]:
    for data in ["iedb", "nod"]:
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
    bash_header="""_this_script=`readlink -f $0`
_this_dir=`dirname $_this_script`

python_env="/usr/prog/sb/mw/python/conda/sb-aegis/bin/python"
qsub_flags="-pe smp 1 -l gpu_card=1 -l gpu_arch=tesla_v100* -l h_rt=12000 -l m_mem_free=64G -cwd -j y -b y "

"""
    this_dir=os.path.dirname(os.path.realpath(__file__))
    best_models = find_best_models()
    with open(os.path.join(this_dir,"evaluation_pipeline.sh"), "w") as f:
        f.write(bash_header)
        for script in ["cd4.py", "maria.py", "nod.py"]:
            f.write("echo %s\n\n" % script)
            for model in best_models.values():
                f.write("echo %(model)s %(data)s \n\n" % model)
                if model["features"] == "seq_only":
                    evaluation_script_params = f"$python_env $_this_dir/../src/mhciipresentation/evaluation/{script} --model_wo_pseudo_path '{model['model']}' --results {LOGS_DIR}/evaluation/{model['features']}_{model['data']}"
                else:
                    evaluation_script_params = f"$python_env $_this_dir/../src/mhciipresentation/evaluation/{script} --model_with_pseudo_path '{model['model']}' --results {LOGS_DIR}/evaluation/{model['features']}_{model['data']}"

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
