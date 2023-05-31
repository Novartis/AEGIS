# -*- coding: utf-8 -*-
"""generate_model_variants.py

This script generates the various configurations of aegis configurations.

"""

import itertools
import os
from pathlib import Path


def write_job(script_contents, path_to_scripts, fname):
    with open(path_to_scripts / fname, "w", encoding="utf-8") as f:
        f.write(script_contents)


def build_job(overrides, path_to_scripts):
    script_cotents = f"""#!/bin/bash -l
##############################
#       Training AEGIS       #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=imm
#SBATCH --cpus-per-task=15
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --time=20-00:00:00
#SBATCH --partition=p.hpcl91
#SBATCH --output=/fs/pool/pool-hartout/Documents/Git/AEGIS/outputs/slurm_outputs/slurm-%j.out
#SBATCH --mail-user=hartout@biochem.mpg.de 
#SBATCH --mail-type=FAIL 
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
    write_job(script_cotents, path_to_scripts, fname)


def main():
    path_to_scripts = (
        Path("/fs/pool/pool-hartout/Documents/Git/AEGIS")
        / "experiments"
        / "variants"
    )
    filelist = [f for f in os.listdir(path_to_scripts) if f.endswith(".sh")]
    for f in filelist:
        os.remove(os.path.join(path_to_scripts, f))

    feature_set = ["seq_only", "seq_mhc"]
    data_source = ["iedb", "iedb_nod", "nod"]
    layers = [2, 4, 8]
    seeds = [0, 1, 2, 3]

    combinations = list(
        itertools.product(feature_set, data_source, layers, seeds)
    )
    for comb in combinations:
        build_job(
            f"dataset.data_source={comb[1]} "
            + f"model.feature_set={comb[0]} "
            + f"seed.seed={comb[3]} "
            + f"model.aegis.n_layers={comb[2]} "
            + f"hydra.run.dir=outputs/variants/{comb[0]}-{comb[1]}-{comb[2]}-{comb[3]}",
            path_to_scripts,
        )


if __name__ == "__main__":
    main()
