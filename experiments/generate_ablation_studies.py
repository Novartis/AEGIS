# -*- coding: utf-8 -*-
"""generate_ablation_studies.py

This script generates the ablation studies for the embedding layer.
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
#SBATCH --job-name=imm_abla
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --time=20-00:00:00
#SBATCH --partition=p.hpcl91
#SBATCH --output=/fs/home/hartout/slurm/slurm-%j.out
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
        / "ablations"
    )
    filelist = [f for f in os.listdir(path_to_scripts) if f.endswith(".sh")]
    for f in filelist:
        os.remove(os.path.join(path_to_scripts, f))

    feature_set = ["seq_mhc"]
    data_source = ["iedb"]
    embedding = [
        "true",
    ]
    all_ones = ["true", "false"]
    seeds = [0, 1, 2, 3]
    abla = ["abla"]
    combinations = list(
        itertools.product(
            feature_set, data_source, embedding, all_ones, seeds, abla
        )
    )
    home_dir = "/fs/home/hartout"
    for comb in combinations:
        build_job(
            f"dataset.data_source={comb[1]} "
            + f"model.feature_set={comb[0]} "
            + f"seed.seed={comb[4]} "
            + f"model.aegis.embedding.dummy_embedding={comb[2]} "
            + f"model.aegis.embedding.all_ones={comb[3]} "
            + f"hydra.run.dir={home_dir}/logs/ablations/{comb[0]}-{comb[1]}-{comb[2]}-{comb[3]}-{comb[4]}-{comb[5]}",
            path_to_scripts,
        )


if __name__ == "__main__":
    main()
