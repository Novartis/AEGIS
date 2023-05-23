# -*- coding: utf-8 -*-
"""submit_variants.py

"""

import os


def main():
    for batch_file in os.listdir(
        "/fs/pool/pool-hartout/Documents/Git/AEGIS/experiments/variants/"
    ):
        os.system(
            "cd /fs/pool/pool-hartout/Documents/Git/AEGIS && sbatch"
            f" /fs/pool/pool-hartout/Documents/Git/AEGIS/experiments/variants/{batch_file}"
        )


if __name__ == "__main__":
    main()
