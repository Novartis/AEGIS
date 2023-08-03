"""generate_inference_scripts.py

In this script, we provide the and slurm commands overrides for the various inference scripts.
"""

import hashlib
import itertools
import os
from pathlib import Path


def write_job(script_contents, path_to_scripts, fname):
    path_to_scripts.mkdir(parents=True, exist_ok=True)
    with open(path_to_scripts / fname, "w", encoding="utf-8") as f:
        f.write(script_contents)


def build_job(script_name, overrides, path_to_scripts):
    script_cotents = f"""#!/bin/bash -l
##############################
#         Inference          #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=imm_inf
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=20-00:00:00
#SBATCH --partition=p.hpcl91
#SBATCH --output=/fs/pool/pool-hartout/Documents/Git/AEGIS/outputs/slurm_outputs/slurm-%j.out
#SBATCH --mail-user=hartout@biochem.mpg.de 
#SBATCH --mail-type=FAIL 
source /fs/home/${"USER"}/.bashrc
cd /fs/pool/pool-hartout/Documents/Git/AEGIS
export CUDA_VISIBLE_DEVICES=0   
/fs/pool/pool-hartout/.conda/envs/aegis/bin/python experiments/evaluation/{script_name} {overrides}

exit 0
    """

    # get hash of overrides
    fname = hashlib.sha256(overrides.encode()).hexdigest()[:8] + ".sh"
    # If fname exists, expand name:
    if (path_to_scripts / fname).exists():
        fname = hashlib.sha256(overrides.encode()).hexdigest()[:16] + ".sh"
        print("Pingeonhole principle hit!")

    write_job(
        script_cotents,
        path_to_scripts,
        fname,
    )


def build_inference_set(
    script_name,
    feature_set_options,
    dataset_options,
    layers,
    seeds,
    path_to_scripts,
):
    for feature_set_option, dataset_option, layer, seed in itertools.product(
        feature_set_options, dataset_options, layers, seeds
    ):
        overrides = f"dataset.data_source={dataset_option} model.feature_set={feature_set_option} model.aegis.n_layers={layer} paths.checkpoint=outputs/variants/{feature_set_option}-{dataset_option}-{layer}-{seed}/checkpoints/last.ckpt hydra.run.dir=outputs/inference/{script_name.split('.')[0]}/{feature_set_option}-{dataset_option}-{layer}-{seed}"
        build_job(script_name, overrides, path_to_scripts)


def main():
    path_to_scripts = (
        Path("/fs/pool/pool-hartout/Documents/Git/AEGIS")
        / "experiments"
        / "inference"
    )
    for dirpath, dirnames, filenames in os.walk(path_to_scripts):
        for filename in filenames:
            if filename.endswith(".sh"):
                os.remove(os.path.join(dirpath, filename))

    # layers = ["4"]
    layers = ["2", "4", "8"]
    seeds = ["0", "1", "2", "3"]
    # cd4
    build_inference_set(
        script_name="cd4.py",
        feature_set_options=["seq_only", "seq_mhc"],
        dataset_options=["iedb_nod", "iedb"],
        layers=layers,
        seeds=seeds,
        path_to_scripts=path_to_scripts / "cd4",
    )

    # hold out
    build_inference_set(
        script_name="hold_out.py",
        feature_set_options=["seq_only", "seq_mhc"],
        dataset_options=["iedb_nod", "iedb"],
        layers=layers,
        seeds=seeds,
        path_to_scripts=path_to_scripts / "hold_out",
    )

    # Maria
    build_inference_set(
        script_name="maria.py",
        feature_set_options=["seq_only", "seq_mhc"],
        dataset_options=["iedb_nod", "iedb"],
        layers=layers,
        seeds=seeds,
        path_to_scripts=path_to_scripts / "maria",
    )

    # NOD
    build_inference_set(
        script_name="nod.py",
        feature_set_options=["seq_only", "seq_mhc"],
        dataset_options=["iedb_nod", "nod", "iedb"],
        layers=layers,
        seeds=seeds,
        path_to_scripts=path_to_scripts / "nod",
    )

    # Stratmann
    build_inference_set(
        script_name="stratmann.py",
        feature_set_options=["seq_only", "seq_mhc"],
        dataset_options=["iedb_nod", "nod"],
        layers=layers,
        seeds=seeds,
        path_to_scripts=path_to_scripts / "stratmann",
    )

    # you
    build_inference_set(
        script_name="you.py",
        feature_set_options=["seq_only", "seq_mhc"],
        dataset_options=["iedb_nod", "iedb"],
        layers=layers,
        seeds=seeds,
        path_to_scripts=path_to_scripts / "you",
    )

    # xu
    build_inference_set(
        script_name="xu.py",
        feature_set_options=["seq_only", "seq_mhc"],
        dataset_options=["iedb_nod", "iedb"],
        layers=layers,
        seeds=seeds,
        path_to_scripts=path_to_scripts / "xu",
    )


if __name__ == "__main__":
    main()
