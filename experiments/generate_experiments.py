# -*- coding: utf-8 -*-
"""generate_experiments.py

This script generates the various configurations of gae experiments.

"""

import hashlib
import os
from pathlib import Path

from pyprojroot import here


def write_job(script_contents, fname):
    with open(
        Path("/fs/pool/pool-hartout/Documents/Git/AEGIS")
        / "experiments"
        / "job_scripts"
        / fname,
        "w",
    ) as f:
        f.write(script_contents)


def build_job(overrides):
    script_cotents = f"""#!/bin/bash -l
##############################
#       Training ZINC        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=imm
#SBATCH --cpus-per-task=19
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=shard:19
#SBATCH --time=20-00:00:00
#SBATCH --partition=p.hpcl91
#SBATCH --output=/fs/pool/pool-hartout/Documents/Git/AEGIS/outputs/slurm_outputs/slurm-%j.out
source /fs/home/${"USER"}/.bashrc
cd /fs/pool/pool-hartout/Documents/Git/AEGIS
export CUDA_VISIBLE_DEVICES=0   
/fs/pool/pool-hartout/.conda/envs/aegis/bin/python experiments/train.py {overrides}

exit 0
    """

    # get hash of overrides
    fname = (
        "_".join(" ".join(overrides.split("=")).split(" ")[1:][::2][:-1])
        + ".sh"
    )
    write_job(script_cotents, fname)


def main():
    path_to_scripts = (
        Path("/fs/pool/pool-hartout/Documents/Git/AEGIS")
        / "experiments"
        / "job_scripts"
    )
    filelist = [f for f in os.listdir(path_to_scripts) if f.endswith(".sh")]
    for f in filelist:
        os.remove(os.path.join(path_to_scripts, f))

    feature_set = ["seq_only", "seq_mhc"]
    data_source = ["iedb", "iedb_nod", "nod"]
    layers = [4, 8]
    seeds = [0, 1, 2]
    for seed in seeds:
        for feat in feature_set:
            for ds in data_source:
                for layer in layers:
                    build_job(
                        f"dataset.data_source={ds} "
                        + f"model.feature_set={feat} "
                        + f"seed.seed={seed} "
                        + f"model.aegis.n_layers={layer} "
                        + f"hydra.run.dir=outputs/{ds}-{feat}-{seed}-{layer}"
                    )


if __name__ == "__main__":
    main()
