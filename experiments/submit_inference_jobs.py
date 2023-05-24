"""submit_inference_jobs.py

Submit inference jobs to the cluster by walking through the inference scripts
"""

import os
from pathlib import Path


def main():
    path_to_scripts = (
        Path("/fs/pool/pool-hartout/Documents/Git/AEGIS")
        / "experiments"
        / "inference"
    )
    for dirpath, _, filenames in os.walk(path_to_scripts):
        for filename in filenames:
            if filename.endswith(".sh"):
                os.system(
                    "cd /fs/pool/pool-hartout/Documents/Git/AEGIS && sbatch"
                    f" {dirpath}/{filename}"
                )


if __name__ == "__main__":
    main()
